$ErrorActionPreference = "Stop"

$OutDir = "C:\Users\ADMIN\Desktop\projects\RL based autonomous interceptor drone\stage-runtime\final_videos"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Generating 8 model videos..."
for ($i=1; $i -le 8; $i++) {
    Write-Host "Rendering stage $i with model..."
    docker compose run --rm stage-runtime python -m stage_runtime --stage $i --model /app/stage-runtime/models/PPO_stage_$($i)_final.zip --out /app/stage-runtime/final_videos
}

Write-Host "All done!"
