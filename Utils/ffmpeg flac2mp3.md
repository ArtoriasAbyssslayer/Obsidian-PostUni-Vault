```pwsh
# Define the target directory for MP3 files
$targetDir = ".\mp3Music"

# Ensure the target directory exists
if (!(Test-Path -Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir
}

# Get all directories in the current directory
$directories = Get-ChildItem -Directory

# Loop through each directory and find FLAC files
foreach ($dir in $directories) {
    # Get all FLAC files in the current directory and subdirectories
    $flacFiles = Get-ChildItem -Path $dir.FullName -Recurse -Filter *.flac

    foreach ($flacFile in $flacFiles) {
        # Define the output MP3 file path in the target directory with the same name as the FLAC file
        $outputFile = Join-Path -Path $targetDir -ChildPath ([System.IO.Path]::ChangeExtension($flacFile.Name, ".mp3"))
        
        # Run the ffmpeg command to convert the FLAC file to MP3
        ffmpeg -i $flacFile.FullName -ab 320k -map_metadata 0 -id3v2_version 3 $outputFile
    }
}

Write-Host "Conversion complete."

```
