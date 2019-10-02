# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files

added_files = collect_data_files('tinycss2')

extra_imports = ['matplotlib', 'reportlab', 'svglib', 'tinycss2']

a = Analysis(['scripts\\run_all.py'],
             pathex=['C:\\Users\\blaur\\Documents\\programming\\schedvisu'],
             binaries=[],
             datas=added_files,
             hiddenimports=extra_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='run_all',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
