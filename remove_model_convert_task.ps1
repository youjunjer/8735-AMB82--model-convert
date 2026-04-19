$ErrorActionPreference = "Stop"

$taskName = "AMB82-TeachableMachine-Convert"

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "Removed scheduled task: $taskName"
} else {
    Write-Output "Scheduled task not found: $taskName"
}
