$datasetRoot = "E:\FreshLens\dataset"
$folders = @(
  "$datasetRoot\raw\apple\fresh",
  "$datasetRoot\raw\apple\rotten",
  "$datasetRoot\raw\banana\fresh",
  "$datasetRoot\raw\banana\rotten",
  "$datasetRoot\raw\orange\fresh",
  "$datasetRoot\raw\orange\rotten",
  "$datasetRoot\raw\tomato\fresh",
  "$datasetRoot\raw\tomato\rotten",
  "$datasetRoot\raw\other\numbers",
  "$datasetRoot\raw\other\objects",
  "$datasetRoot\raw\other\other_food"
)
foreach ($folder in $folders) {
  New-Item -ItemType Directory -Force -Path $folder | Out-Null
}
Write-Host "Đã tạo cấu trúc dataset tại $datasetRoot"
Write-Host 'Trong PowerShell hiện tại, chạy: $env:FRESHLENS_DATASET_ROOT = "E:\FreshLens\dataset"'

