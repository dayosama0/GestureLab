import io
import struct
import zipfile
import zlib

from PIL import Image
import pytest
from ml import hagrid_expand as h


def test_crop_matching_geometry_and_gray_padding():
    image=Image.new('RGB',(1000,800),'white')
    actual=h.crop_box(image,[.2,.25,.2,.25])
    assert all(abs(a-b)<=1 for a,b in zip(actual,(170,170,430,430)))
    for box in [[0,0,0,.4],[.9,0,.2,.3],[0,0,float('nan'),.3],[0,0,.01,.01]]:
        with pytest.raises(ValueError):
            h.crop_box(image,box)


def test_bounded_member_crc_and_data(monkeypatch):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('nested/test.txt',b'actual payload'*100)
    data=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        source=archive.infolist()[0]
    remote=object.__new__(h.RemoteArchive)
    remote.url='test://bounded';remote.size=len(data)
    entry=h.Entry(source.filename,source.header_offset,source.compress_size,source.file_size,source.compress_type,source.CRC)
    calls=[]
    monkeypatch.setattr(h,'range_read',lambda url,start,end,total:(calls.append((start,end)) or data[start:end+1]))
    assert remote.member(entry)==b'actual payload'*100
    assert calls==[(0,len(data)-1)]
    entry.crc^=1
    with pytest.raises(ValueError,match='CRC'):
        remote.member(entry)


def test_range_server_ignoring_bounds_rejected(monkeypatch):
    class Response:
        status_code=200
        headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_raw(self,*args,**kwargs):raise AssertionError('Must reject before reading full body')
    monkeypatch.setattr(h.HTTP,'stream',lambda *a,**kw:Response())
    with pytest.raises(ValueError,match='honor'):
        h.range_read('https://source.invalid/huge.zip',0,29)
