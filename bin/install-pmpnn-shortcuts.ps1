$ErrorActionPreference = 'Stop'

# Resolve the clone from this script so the shortcuts survive any current directory.
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$assetsDir = Join-Path $repoRoot 'assets'
$iconPath = Join-Path $assetsDir 'proteinmpnn.ico'
New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class NativeIcon {
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern bool DestroyIcon(IntPtr handle);
}
'@

# A compact, high-contrast folded-ribbon mark designed to remain readable at
# Start Menu and taskbar sizes. The ICO is generated locally, avoiding assets
# or setup outside this repository.
$size = 256
$bitmap = [System.Drawing.Bitmap]::new($size, $size)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$graphics.Clear([System.Drawing.Color]::Transparent)

$tile = [System.Drawing.RectangleF]::new(8, 8, 240, 240)
$tilePath = [System.Drawing.Drawing2D.GraphicsPath]::new()
$radius = 54
$tilePath.AddArc($tile.X, $tile.Y, $radius, $radius, 180, 90)
$tilePath.AddArc($tile.Right - $radius, $tile.Y, $radius, $radius, 270, 90)
$tilePath.AddArc($tile.Right - $radius, $tile.Bottom - $radius, $radius, $radius, 0, 90)
$tilePath.AddArc($tile.X, $tile.Bottom - $radius, $radius, $radius, 90, 90)
$tilePath.CloseFigure()
$background = [System.Drawing.Drawing2D.LinearGradientBrush]::new($tile, [System.Drawing.Color]::FromArgb(20, 37, 72), [System.Drawing.Color]::FromArgb(17, 96, 170), 45)
$graphics.FillPath($background, $tilePath)

# Folded protein-ribbon paths: a pale backbone plus teal and blue highlights.
$backbone = [System.Drawing.Drawing2D.GraphicsPath]::new()
$backbone.AddBezier(48, 156, 73, 72, 103, 74, 125, 128)
$backbone.AddBezier(125, 128, 149, 183, 178, 183, 208, 92)
$graphics.DrawPath([System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(214, 236, 250), 26), $backbone)
$graphics.DrawPath([System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(75, 217, 217), 11), $backbone)

$ribbon = [System.Drawing.Drawing2D.GraphicsPath]::new()
$ribbon.AddBezier(48, 96, 82, 181, 108, 179, 129, 127)
$ribbon.AddBezier(129, 127, 153, 68, 180, 74, 210, 160)
$graphics.DrawPath([System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(95, 145, 255), 18), $ribbon)
$graphics.DrawPath([System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(190, 235, 255), 5), $ribbon)

foreach ($point in @([System.Drawing.PointF]::new(48, 156), [System.Drawing.PointF]::new(125, 128), [System.Drawing.PointF]::new(208, 92), [System.Drawing.PointF]::new(48, 96), [System.Drawing.PointF]::new(210, 160))) {
    $graphics.FillEllipse([System.Drawing.Brushes]::White, $point.X - 6, $point.Y - 6, 12, 12)
}
$graphics.Dispose()
$iconHandle = $bitmap.GetHicon()
try {
    $icon = [System.Drawing.Icon]::FromHandle($iconHandle)
    $stream = [System.IO.File]::Create($iconPath)
    try { $icon.Save($stream) } finally { $stream.Dispose(); $icon.Dispose() }
} finally {
    [NativeIcon]::DestroyIcon($iconHandle) | Out-Null
    $bitmap.Dispose()
}

$shell = New-Object -ComObject WScript.Shell
$launchTarget = Join-Path $repoRoot 'bin\pmpnn-gui.bat'
$shortcutLocations = @(
    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'ProteinMPNN Studio.lnk'),
    (Join-Path (Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs') 'ProteinMPNN Studio.lnk')
)
foreach ($shortcutPath in $shortcutLocations) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $launchTarget
    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.IconLocation = "$iconPath,0"
    $shortcut.Description = 'ProteinMPNN Studio — structure-guided protein sequence design'
    $shortcut.Save()
}

Write-Host 'ProteinMPNN Studio shortcuts were added to your Desktop and Start Menu.' -ForegroundColor Cyan
