; Inno Setup script for Eyenips — wraps the PyInstaller one-dir build into a
; Windows installer.  Build the exe first (pyinstaller eyenips.spec), then open
; this in Inno Setup (https://jrsoftware.org/isdl.php) and click Compile, or:
;     iscc installer\eyenips.iss
;
; Per-user install (no admin prompt, no SmartScreen elevation) into LocalAppData
; so the app can drop content/effect updates into its own folder WITHOUT admin —
; matching Eyenips' "loose, updatable effects" design. User data (presets,
; sessions, lab kits) lives separately in %USERPROFILE%\.eyenips and is never
; touched by install/uninstall/update.

#define MyAppName "Eyenips"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Eyenips"
#define MyAppExeName "Eyenips.exe"

[Setup]
AppId={{A9E3F2B1-1E5C-4C2E-9B7A-EYENIPS00001}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist_installer
OutputBaseFilename=Eyenips-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; SetupIconFile=installer\eyenips.ico
; UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; the entire PyInstaller one-dir output (exe + _internal + loose effects/ + presets/)
Source: "..\dist\{#MyAppName}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
