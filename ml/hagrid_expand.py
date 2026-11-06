"""Acquire audited, participant-separated HaGRID ROIs using bounded ZIP ranges.

Only selected compressed image members are transferred, never whole 40 GB ZIPs.
Original data and validation/test splits are not modified. Source frames remain
in memory; locally cached crops and their provenance support reproducibility.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import struct
import threading
import time
import urllib.request
import zipfile
import zlib

import httpx
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'hagrid_cache'
OUTPUT = ROOT / 'data' / 'expanded'
UPSTREAM = 'https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/'
ANNOTATIONS = 'https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip'
LABELS = {'fist': 'fist', 'palm': 'open_palm', 'peace': 'peace', 'like': 'thumbs_up', 'peace_inverted': 'peace'}
DOWNLOAD_LOCK = threading.Lock()
DOWNLOADED = 0
BUDGET = 8 * 1024**3
HTTP = httpx.Client(timeout=90,limits=httpx.Limits(max_connections=6,max_keepalive_connections=6))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def range_read(url, start, end, total=None):
    """Reject servers ignoring Range so no unbounded image ZIP is downloaded."""
    global DOWNLOADED
    if start < 0 or end < start or end-start+1 > 100*1024**2:
        raise ValueError('Invalid or excessive requested range')
    for attempt in range(4):
        try:
            with HTTP.stream('GET',url,headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'GestureLab-research/1.0'}) as response:
                content_range = response.headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                if response.status_code != 206 or not match:
                    raise ValueError('Server did not honor bounded byte Range')
                actual_start, actual_end, actual_total = map(int, match.groups())
                if actual_start != start or actual_end != end or (total is not None and actual_total != total):
                    raise ValueError('Returned byte Range does not match request/source size')
                chunks=[]
                received=0
                for chunk in response.iter_raw(chunk_size=65536):
                    received+=len(chunk)
                    if received>end-start+1:
                        raise ValueError('Oversized Range response')
                    chunks.append(chunk)
                result=b''.join(chunks)
                if len(result) != end-start+1:
                    raise ValueError('Incomplete or oversized Range response')
            with DOWNLOAD_LOCK:
                DOWNLOADED += len(result)
                if DOWNLOADED > BUDGET:
                    raise RuntimeError('Configured download byte budget exceeded')
            return result
        except (OSError, TimeoutError, httpx.HTTPError) as exc:
            if attempt == 3:
                raise
            print(f'HTTP retry {attempt+1}: {type(exc).__name__}', flush=True)
            time.sleep(1+attempt*2)


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
            if response.status != 206 or not match:
                raise ValueError('ZIP source must support HTTP Range')
            self.size = int(match.group(1))
            self.etag = response.headers.get('ETag')
            response.read(2)

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if not 0 <= self.pos <= self.size:
            raise ValueError('Seek outside remote archive')
        return self.pos

    def read(self, count=-1):
        if count < 0:
            count = self.size-self.pos
        count = min(count, self.size-self.pos)
        if not count:
            return b''
        data = range_read(self.url, self.pos, self.pos+count-1, self.size)
        self.pos += len(data)
        return data


@dataclass
class Entry:
    filename: str
    offset: int
    compressed: int
    uncompressed: int
    compression: int
    crc: int


class RemoteArchive:
    def __init__(self, url, cache_name):
        self.url = url
        reader = RangeFile(url)
        self.size = reader.size
        cache = CACHE / f'{cache_name}_index.json'
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved['size'] != self.size or saved['etag'] != reader.etag:
                raise ValueError('Upstream archive changed; explicit cache review required')
            self.entries = {e['filename']: Entry(**e) for e in saved['entries']}
        else:
            with zipfile.ZipFile(reader) as archive:
                entries = [Entry(e.filename, e.header_offset, e.compress_size, e.file_size, e.compress_type, e.CRC) for e in archive.infolist() if not e.is_dir()]
            self.entries = {e.filename: e for e in entries}
            cache.write_text(json.dumps({'url':url, 'size':self.size, 'etag':reader.etag, 'entries':[asdict(e) for e in entries]}))

    def member(self, entry):
        if isinstance(entry, str):
            entry = self.entries[entry]
        if entry.uncompressed > 150*1024**2 or entry.compressed > 30*1024**2:
            raise ValueError('Unexpectedly large ZIP member')
        # Read header + payload in one request; the bounded allowance covers ZIP64 extras.
        stop = min(self.size-1, entry.offset+30+len(entry.filename.encode('utf-8'))+512+entry.compressed-1)
        data = range_read(self.url, entry.offset, stop, self.size)
        values = struct.unpack('<4s5H3I2H', data[:30])
        if values[0] != b'PK\x03\x04':
            raise ValueError('Invalid ZIP local header')
        start = 30+values[-2]+values[-1]
        compressed = data[start:start+entry.compressed]
        if len(compressed) != entry.compressed:
            raise ValueError('ZIP extra header exceeded bounded allowance')
        if entry.compression == zipfile.ZIP_DEFLATED:
            result = zlib.decompress(compressed, -15)
        elif entry.compression == zipfile.ZIP_STORED:
            result = compressed
        else:
            raise ValueError('Unsupported compression')
        if len(result) != entry.uncompressed or zlib.crc32(result) & 0xffffffff != entry.crc:
            raise ValueError('ZIP member length/CRC check failed')
        return result


def dhash(image):
    pixels = np.asarray(image.convert('L').resize((9, 8)))
    return sum(int(v) << i for i,v in enumerate((pixels[:,1:] > pixels[:,:-1]).flat))


def crop_box(image, normalized, margin=0.15):
    if len(normalized) != 4 or not all(math.isfinite(v) for v in normalized):
        raise ValueError('Invalid normalized bounding box')
    x,y,w,h = normalized
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > 1.001 or y+h > 1.001:
        raise ValueError('Bounding box lies outside source')
    left=max(0, math.floor((x-margin*w)*image.width))
    top=max(0, math.floor((y-margin*h)*image.height))
    right=min(image.width, math.ceil((x+w+margin*w)*image.width))
    bottom=min(image.height, math.ceil((y+h+margin*h)*image.height))
    if min(w*image.width, h*image.height) < 80:
        raise ValueError('Hand box is too small for reliable task crop')
    return (left,top,right,bottom)


def load_annotations(archive, split, gesture):
    path = CACHE / f'annotations_{split}_{gesture}.json'
    if not path.exists():
        content = archive.member(f'annotations/{split}/{gesture}.json')
        path.write_bytes(content)
    payload=path.read_bytes()
    entry=archive.entries[f'annotations/{split}/{gesture}.json']
    if len(payload)!=entry.uncompressed or zlib.crc32(payload)&0xffffffff!=entry.crc:
        raise ValueError('Cached annotations do not match official archive CRC/size')
    return json.loads(payload)


def original_inventory():
    records=[]
    for path in sorted((ROOT/'data'/'ready_dataset').rglob('*.jpg')):
        with Image.open(path) as image:
            perceptual=dhash(image)
        split,label = path.parts[-3:-1]
        records.append({'path':path.relative_to(ROOT).as_posix(),'label':label,'split':split,
                        'source_id':path.stem,'sha256':sha(path.read_bytes()),'dhash':perceptual})
    if len(records)!=400:
        raise ValueError('Expected original 400-image dataset')
    return records


def build(args):
    global BUDGET
    BUDGET=int(args.budget_gb*1024**3)
    CACHE.mkdir(parents=True,exist_ok=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'manifest.json').exists():
        raise FileExistsError('Expanded manifest already exists; preserve it as an immutable dataset version')
    started=time.monotonic()
    license_path=CACHE/'HaGRID_license_en_us.pdf'
    if not license_path.exists():
        with urllib.request.urlopen('https://raw.githubusercontent.com/hukenovs/hagrid/master/license/en_us.pdf',timeout=60) as response:
            license_path.write_bytes(response.read(2*1024**2))
    original=original_inventory()
    original_ids={r['source_id'] for r in original}
    original_hashes={r['sha256'] for r in original}
    holdout_hashes=[r['dhash'] for r in original if r['split']!='train']
    annotation_archive=RemoteArchive(ANNOTATIONS,'annotation')
    metadata={}
    keys=[(s,g) for s in ['train','val','test'] for g in LABELS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key,records in zip(keys,pool.map(lambda key:load_annotations(annotation_archive,*key),keys)):
            metadata[key]=records
            print(f'Annotations {key}: {len(records)} source images',flush=True)
    source_metadata={source:record for records in metadata.values() for source,record in records.items()}
    original_users={source_metadata[source]['user_id'] for source in original_ids if source in source_metadata}
    unresolved=sorted(original_ids-source_metadata.keys())
    print(f'Original metadata matches: {400-len(unresolved)}/400; excluded participants: {len(original_users)}',flush=True)
    original_user_sets={s:{source_metadata[r['source_id']]['user_id'] for r in original if r['split']==s and r['source_id'] in source_metadata} for s in ['train','val','test']}
    original_user_overlap={a+'_'+b:len(original_user_sets[a]&original_user_sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    # Verify upstream split membership for ALL eligible sources, not just selected rows.
    user_splits={}
    for (split,gesture),records in metadata.items():
        for source,record in records.items():
            user=record['user_id']
            if user in original_users:
                continue
            previous=user_splits.setdefault(user,split)
            if previous!=split:
                raise ValueError('Upstream participant crosses train/val/test; explicit subject regrouping needed')
    archives={}
    for gesture in LABELS:
        archives[gesture]=RemoteArchive(UPSTREAM+gesture+'.zip',gesture)
        print(f'Indexed {gesture}: {len(archives[gesture].entries)} image members; no full archive downloaded',flush=True)
    member_index={g:{Path(name).stem:entry for name,entry in archive.entries.items() if name.lower().endswith(('.jpg','.jpeg'))} for g,archive in archives.items()}
    log=OUTPUT/'records.jsonl'
    records=[]
    if log.exists():
        for line in log.read_text().splitlines():
            record=json.loads(line)
            path=ROOT/record['path']
            if not path.exists() or sha(path.read_bytes())!=record['sha256']:
                raise ValueError('Cached crop integrity mismatch')
            records.append(record)
    seen_sources={r['source_id'] for r in records}
    seen_crops={r['sha256'] for r in records}
    accepted_perceptual=[(r['split'],r['dhash']) for r in records]
    rejections=Counter()
    targets={'train':args.train,'val':args.val,'test':args.test}
    append_lock=threading.Lock()

    def fetch_crop(candidate):
        gesture,split,source,annotation=candidate
        try:
            raw=archives[gesture].member(member_index[gesture][source])
            source_hash=sha(raw)
            if source_hash in original_hashes:
                return None,'source_hash_overlap'
            with Image.open(io.BytesIO(raw)) as full:
                full.load()
                if full.size[0]*full.size[1]>32_000_000:
                    return None,'oversized_source'
                image=ImageOps.exif_transpose(full).convert('RGB')
            matches=[box for box,label in zip(annotation['bboxes'],annotation['labels']) if label==gesture]
            if not matches:
                return None,'no_matching_hand_box'
            normalized=max(matches,key=lambda b:b[2]*b[3])
            box=crop_box(image,normalized)
            crop=ImageOps.pad(image.crop(box),(320,320),method=Image.Resampling.LANCZOS,color=(128,128,128))
            perceptual=dhash(crop)
            original_compare=holdout_hashes if split=='train' else [r['dhash'] for r in original]
            if any((perceptual^other).bit_count()<=3 for other in original_compare):
                return None,'holdout_perceptual_candidate'
            buffer=io.BytesIO();crop.save(buffer,format='PNG');content=buffer.getvalue()
            destination=OUTPUT/split/LABELS[gesture]/(source+'.png')
            destination.parent.mkdir(parents=True,exist_ok=True)
            record={'path':destination.relative_to(ROOT).as_posix(),'label':LABELS[gesture],
                    'split':split,'source_id':source,'user_id':annotation['user_id'],
                    'origin':'HaGRID official bounded ZIP member', 'upstream_gesture':gesture,
                    'source_sha256':source_hash,'sha256':sha(content),'dhash':perceptual,
                    'bbox_normalized_xywh':normalized,'crop_box_pixels':box,
                    'source_size':image.size,'bbox_margin_fraction':0.15,'stored_size':320,
                    'source_archive':archives[gesture].url,'source_member':member_index[gesture][source].filename}
            with append_lock:
                if record['sha256'] in seen_crops:
                    return None,'duplicate_crop_hash'
                if any(split!=other_split and (perceptual^other_hash).bit_count()<=3 for other_split,other_hash in accepted_perceptual):
                    return None,'new_cross_split_perceptual_candidate'
                seen_crops.add(record['sha256'])
                accepted_perceptual.append((split,perceptual))
                destination.write_bytes(content)
                with log.open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(record)+'\n')
            return record,None
        except (ValueError,OSError,KeyError) as exc:
            return None,f'failed_{type(exc).__name__}'

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for gesture,label in LABELS.items():
            for split in ['train','val','test']:
                target=round(targets[split]*0.25) if gesture=='peace_inverted' else targets[split]-round(targets[split]*0.25) if gesture=='peace' else targets[split]
                count=sum(r['upstream_gesture']==gesture and r['split']==split for r in records)
                candidates=[]
                used_users={r['user_id'] for r in records if r['upstream_gesture']==gesture and r['split']==split}
                items=sorted(metadata[split,gesture].items())
                random.Random(42).shuffle(items)
                for source,annotation in items:
                    user=annotation['user_id']
                    if not re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',source):
                        continue
                    if source in original_ids or source in seen_sources or source not in member_index[gesture] or user in original_users or user in used_users:
                        continue
                    used_users.add(user)
                    candidates.append((gesture,split,source,annotation))
                position=0
                while count<target:
                    batch=candidates[position:position+min(48,target-count)]
                    if not batch:
                        raise RuntimeError(f'Insufficient eligible unique participants for {gesture}/{split}: {count}')
                    position+=len(batch)
                    for record,error in pool.map(fetch_crop,batch):
                        if record:
                            records.append(record);seen_sources.add(record['source_id']);count+=1
                        else:
                            rejections[error]+=1
                    print(f'{gesture}/{split}: {count}/{target}; downloaded={DOWNLOADED/1024**2:.1f}MiB; elapsed={time.monotonic()-started:.1f}s',flush=True)
    # Full-crop cross-split perceptual screen; conservative rejection does not establish subject independence itself.
    new_hashes=defaultdict(list)
    for record in records:
        new_hashes[record['split']].append(record)
    near=[]
    for a,b in [('train','val'),('train','test'),('val','test')]:
        for left in new_hashes[a]:
            for right in new_hashes[b]:
                distance=(left['dhash']^right['dhash']).bit_count()
                if distance<=3:
                    near.append({'a':left['source_id'],'b':right['source_id'],'distance':distance})
    original_records=[]
    for record in original:
        if record['split']=='train':
            item=dict(record)
            item['user_id']=source_metadata.get(record['source_id'],{}).get('user_id')
            item['origin']='Original Assignment1 supplied fixed train split'
            original_records.append(item)
    all_records=original_records+records
    counts={split:dict(Counter(r['label'] for r in all_records if r['split']==split)) for split in ['train','val','test']}
    selected_user_sets={s:{r['user_id'] for r in records if r['split']==s} for s in ['train','val','test']}
    if any(original_users & users for users in selected_user_sets.values()):
        raise ValueError('Recovered original participant leaked into added data')
    if any(selected_user_sets[a]&selected_user_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Selected added participants cross splits')
    if len({r['source_id'] for r in all_records})!=len(all_records):
        raise ValueError('Repeated source IDs in expanded manifest')
    if any(counts[s].get(label)!=targets[s]+(70 if s=='train' else 0) for s in targets for label in set(LABELS.values())):
        raise ValueError('Expanded logical-class counts do not match fixed balanced targets')
    audit={'ready':True,'seed':42,'counts':counts,'additional_counts':{s:dict(Counter(r['label'] for r in records if r['split']==s)) for s in targets},
           'original_images':len(original),'original_annotations_matched':400-len(unresolved),
           'original_unresolved_source_ids':unresolved,'excluded_original_participants':len(original_users),
           'unresolved_original_source_ids':unresolved,'original_split_participant_overlaps':original_user_overlap,
           'all_original_participants_excluded':not unresolved,
           'additional_participants_disjoint':True,'original_val_test_preserved':True,
           'additional_unique_participants':{s:len(users) for s,users in selected_user_sets.items()},
           'exact_source_overlap':0,'crop_hash_duplicates':0,'original_holdout_perceptual_candidates_retained':0,
           'new_cross_split_perceptual_candidates':near,'rejected':dict(rejections),
           'perceptual_overlap_review_complete':not near,
           'download_bytes_this_run':DOWNLOADED,'download_budget_bytes':BUDGET,
           'elapsed_seconds':time.monotonic()-started,'crop_policy':'matching official hand bbox;15% each-side margin; gray128 letterbox320 RGB; no training augmentation saved',
           'source':'https://github.com/hukenovs/hagrid','annotations_url':ANNOTATIONS,
           'peace_semantics':'Logicalpeace includes75% officialpeace and25% officialpeace_inverted in additional splits; sourcegesture retained perrecord',
           'license_url':'https://github.com/hukenovs/hagrid/tree/master/license',
           'cached_license':license_path.relative_to(ROOT).as_posix(),'cached_license_sha256':sha(license_path.read_bytes()),
           'license_note':'HaGRID custom public license with attribution/conditions, derived from Attribution-ShareAlike4.0; license explicitly says it is not a Creative Commons license. Official terms cached locally; original participant consent not independently verified.',
           'split_policy':'Original70/class train retained; original val/test immutable and not in new manifest; extra official subject-disjoint splits; exclude recovered original participants; one candidate/user/upstreamgesture',
           'limitations':'Original subject exclusion is verified only for matched source IDs; perceptual hashes are screening, not proof against semantic near-duplicates.'}
    # Near-duplicate findings are an auditable flag; training loader can reject if policy demands.
    manifest={'ready':True,'metadata':audit,'counts':counts,'records':sorted(all_records,key=lambda r:(r['split'],r['label'],r['source_id']))}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (ROOT/'artifacts'/'expansion_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('READY '+json.dumps({k:v for k,v in audit.items() if k not in ['new_cross_split_perceptual_candidates','original_unresolved_source_ids']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=int,default=500)
    parser.add_argument('--val',type=int,default=100)
    parser.add_argument('--test',type=int,default=100)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--budget-gb',type=float,default=8)
    args=parser.parse_args()
    if min(args.train,args.val,args.test,args.workers)<1:
        parser.error('Counts and workers must be positive')
    try:
        build(args)
    finally:
        HTTP.close()


if __name__=='__main__':
    main()
