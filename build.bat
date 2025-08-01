pyinstaller --noconfirm --onedir --console ^
--icon "C:\Users\hungnm\Downloads\image (2).ico" ^
--name "ALPRFAST" ^
--add-binary "D:\jkhung\work\vetc\github\ANPR_YoloV11_VietOCR\mydll\*.dll;." ^
--add-data "D:\jkhung\work\vetc\github\ANPR_YoloV11_VietOCR\config.json;." ^
"D:\jkhung\work\vetc\github\ANPR_YoloV11_VietOCR\ocr_server_fastanpr.py"
copy "dist\ALPRFAST\_internal\config.json" "dist\ALPRFAST\config.json"