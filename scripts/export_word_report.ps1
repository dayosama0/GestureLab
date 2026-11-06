$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$inputDoc = Join-Path $root 'report/report.docx'
$outputPdf = Join-Path $root 'report/report.pdf'
$wordApp = $null
$openedDoc = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $openedDoc = $wordApp.Documents.Open($inputDoc, $false, $false)
    $openedDoc.Fields.Update() | Out-Null
    $openedDoc.Repaginate()
    $pages = $openedDoc.ComputeStatistics(2)
    $openedDoc.Save()
    $openedDoc.ExportAsFixedFormat($outputPdf, 17)
    Write-Output "Native Word-compatible export completed: $pages pages; $outputPdf"
} finally {
    if ($openedDoc) { $openedDoc.Close(0) }
    if ($wordApp) {
        try { $wordApp.Quit() }
        catch [System.Runtime.InteropServices.COMException] {
            # WPS's Word COM adapter can exit when its last document closes.
            if ($_.Exception.HResult -ne -2147023174) { throw }
        }
    }
}
