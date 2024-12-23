# Test the utils module

# Imports
import pathlib
import pytest
from . import utils

#
# Test: SafeOpenForWrite
#

def test_invalid_mode_raises(tmp_path: pathlib.Path):
	# Ensure that providing a mode without 'w' raises a ValueError
	file_path = tmp_path / 'dummy.txt'
	with pytest.raises(ValueError, match="File opening mode must be a truncating write mode"):
		with utils.SafeOpenForWrite(file_path, mode='r'):
			pass

def test_text_write_no_rstack(tmp_path: pathlib.Path):
	# Test writing text data with no RevertStack
	file_path = tmp_path / 'testfile.txt'
	file_content = "Hello, world!"
	assert not file_path.exists()
	with utils.SafeOpenForWrite(file_path, mode='w') as file:
		file.write(file_content)
	assert file_path.exists()
	assert file_path.read_text(encoding='utf-8') == file_content
	assert list(tmp_path.iterdir()) == [file_path]

def test_text_write_existing_file_no_rstack(tmp_path: pathlib.Path):
	# Test overwriting an existing file with no RevertStack
	file_path = tmp_path / 'testfile.txt'
	file_content_old = "Original content"
	file_content_new = "New content"
	file_path.write_text(file_content_old, encoding='utf-8')
	with utils.SafeOpenForWrite(file_path, mode='w') as file:
		file.write(file_content_new)
	assert file_path.read_text(encoding='utf-8') == file_content_new
	assert list(tmp_path.iterdir()) == [file_path]

def test_text_write_rstack_backup(tmp_path: pathlib.Path):
	# Test that a backup is created if a RevertStack is provided, and that it can be reverted
	file_path = tmp_path / 'testfile.txt'
	file_content_old = "Original content"
	file_content_new = "New content"
	file_path.write_text(file_content_old, encoding='utf-8')
	assert file_path.read_text(encoding='utf-8') == file_content_old
	try:
		with utils.RevertStack() as rstack:
			with utils.SafeOpenForWrite(file_path, mode='w', rstack=rstack) as file:
				file.write(file_content_new)
			assert file_path.read_text(encoding='utf-8') == file_content_new
			raise RuntimeError("Triggered revert")  # Raise an exception so that the revert stack triggers a revert
	except RuntimeError as e:
		if not (len(e.args) == 1 and e.args[0] == "Triggered revert"):
			raise
	assert file_path.read_text(encoding='utf-8') == file_content_old
	assert list(tmp_path.iterdir()) == [file_path]

def test_text_write_rstack_no_existing_file(tmp_path: pathlib.Path):
	# Test using SafeOpenForWrite with RevertStack when the file doesn't exist initially
	# No backup should be created, but reverting should remove the newly created file
	file_path = tmp_path / 'newfile.txt'
	file_content = "Brand new file content"
	assert not file_path.exists()
	try:
		with utils.RevertStack() as rstack:
			with utils.SafeOpenForWrite(file_path, mode='w', rstack=rstack) as file:
				file.write(file_content)
			assert file_path.exists()
			assert file_path.read_text(encoding='utf-8') == file_content
			raise RuntimeError('Triggered revert')
	except RuntimeError as e:
		if not (len(e.args) == 1 and e.args[0] == 'Triggered revert'):
			raise
	assert not file_path.exists()
	assert not list(tmp_path.iterdir())

def test_text_write_exception_inside_block(tmp_path: pathlib.Path):
	# Test an exception inside the write with-block
	file_path = tmp_path / 'testfile.txt'
	file_content_old = "Original content"
	file_content_new = "New content"
	file_path.write_text(file_content_old, encoding='utf-8')
	assert file_path.read_text(encoding='utf-8') == file_content_old
	try:
		with utils.SafeOpenForWrite(file_path, mode='w') as file:
			file.write(file_content_new)
			raise RuntimeError('Triggered revert')
	except RuntimeError as e:
		if not (len(e.args) == 1 and e.args[0] == 'Triggered revert'):
			raise
	assert file_path.read_text(encoding='utf-8') == file_content_old
	assert list(tmp_path.iterdir()) == [file_path]

def test_binary_write_no_rstack(tmp_path: pathlib.Path):
	# Test binary mode writing to ensure 'b' doesn't force any encoding/newline usage
	file_path = tmp_path / 'testfile.bin'
	file_content = b'\x00\xFF\x10\x80'
	with utils.SafeOpenForWrite(file_path, mode='bw') as file:
		file.write(file_content)
	assert file_path.read_bytes() == file_content
	assert list(tmp_path.iterdir()) == [file_path]

def test_binary_write_existing_file_with_rstack(tmp_path: pathlib.Path):
	# Test overwriting an existing binary file with a backup revert using RevertStack
	file_path = tmp_path / 'binary.bin'
	file_content_old = b'ABC123'
	file_content_new = b'XYZ987654'
	file_path.write_bytes(file_content_old)
	assert file_path.read_bytes() == file_content_old
	try:
		with utils.RevertStack() as rstack:
			with utils.SafeOpenForWrite(file_path, mode='wb', rstack=rstack) as file:
				file.write(file_content_new)
			assert file_path.read_bytes() == file_content_new
			raise RuntimeError('Triggered revert')
	except RuntimeError as e:
		if not (len(e.args) == 1 and e.args[0] == 'Triggered revert'):
			raise
	assert file_path.read_bytes() == file_content_old
	assert list(tmp_path.iterdir()) == [file_path]

def test_backup_affix_custom(tmp_path: pathlib.Path):
	# Test providing a custom backup affix
	file_path = tmp_path / 'testfile.txt'
	file_content_old = "Original\ncontent"
	file_content_new = "New\ncontent"
	file_path.write_text(file_content_old, encoding='utf-8')
	backup_affix = utils.Affix(prefix='myBackup_', suffix='.old')
	try:
		with utils.RevertStack() as rstack:
			with utils.SafeOpenForWrite(file_path, rstack=rstack, backup_affix=backup_affix) as file:
				file.write(file_content_new)
			assert file_path.read_text(encoding='utf-8') == file_content_new
			raise RuntimeError('Triggered revert')
	except RuntimeError as e:
		if not (len(e.args) == 1 and e.args[0] == 'Triggered revert'):
			raise
	assert file_path.read_text(encoding='utf-8') == file_content_old
	assert list(tmp_path.iterdir()) == [file_path]
# EOF
