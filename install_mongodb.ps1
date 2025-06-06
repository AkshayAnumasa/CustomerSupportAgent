# Create directories for MongoDB
$mongoDbPath = "C:\data\db"
New-Item -ItemType Directory -Force -Path $mongoDbPath

# Download MongoDB installers
$mongoDbUrl = "https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-6.0.8-signed.msi"
$installerPath = "$env:TEMP\mongodb.msi"
Invoke-WebRequest -Uri $mongoDbUrl -OutFile $installerPath

# Install MongoDB
Start-Process msiexec.exe -ArgumentList "/i `"$installerPath`" /quiet ADDLOCAL=`"ServerService,Router,Client,Shell,MonitoringTools,ImportExportTools`" " -Wait

# Add MongoDB to Path
$env:Path += ";C:\Program Files\MongoDB\Server\6.0\bin"
[Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)

# Create MongoDB service
mongod --install --serviceName "MongoDB" --serviceDisplayName "MongoDB" --dbpath "C:\data\db"

# Start MongoDB service
Start-Service MongoDB
