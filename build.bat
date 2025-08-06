pyinstaller --noconfirm --onedir --console ^
--icon ".\mydll\anpr.ico" ^
--name "ALPRFAST" ^
--add-binary ".\mydll\*.dll;." ^
--add-data ".\config.json;." ^
--add-data ".\run.bat;." ^
".\ocr_server_fastanpr_combine.py"
copy "dist\ALPRFAST\_internal\config.json" "dist\ALPRFAST\config.json"
copy "dist\ALPRFAST\_internal\run.bat" "dist\ALPRFAST\run.bat"