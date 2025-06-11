#!/usr/bin/env python3
"""
File renaming script for Discord Bot Translator refactoring.

This script renames files for consistency and updates all import statements.
"""
import os
import re
import shutil
from pathlib import Path

# Define the file renaming map
RENAME_MAP = {
    'custom_sink.py': 'audio_sink.py',
    'custom_sink_core.py': 'audio_sink_core.py',
    'transcription.py': 'transcription_service.py',
    'transcription_core.py': 'transcription_engine.py',
    'translator.py': 'translation_service.py',
    'translator_core.py': 'translation_engine.py',
    'audio_utils.py': 'audio_processing_utils.py',
    'utils.py': 'translation_utils.py'
}

# Import update patterns
IMPORT_PATTERNS = [
    # Direct imports
    (r'\bfrom custom_sink import\b', 'from audio_sink import'),
    (r'\bfrom custom_sink_core import\b', 'from audio_sink_core import'),
    (r'\bfrom transcription import\b', 'from transcription_service import'),
    (r'\bfrom transcription_core import\b', 'from transcription_engine import'),
    (r'\bfrom translator import\b', 'from translation_service import'),
    (r'\bfrom translator_core import\b', 'from translation_engine import'),
    (r'\bfrom audio_utils import\b', 'from audio_processing_utils import'),
    (r'\bfrom utils import\b', 'from translation_utils import'),
    
    # Module imports
    (r'\bimport custom_sink\b', 'import audio_sink'),
    (r'\bimport custom_sink_core\b', 'import audio_sink_core'),
    (r'\bimport transcription\b', 'import transcription_service'),
    (r'\bimport transcription_core\b', 'import transcription_engine'),
    (r'\bimport translator\b', 'import translation_service'),
    (r'\bimport translator_core\b', 'import translation_engine'),
    (r'\bimport audio_utils\b', 'import audio_processing_utils'),
    (r'\bimport utils\b', 'import translation_utils'),
    
    # Qualified imports (as module references)
    (r'\bcustom_sink\.', 'audio_sink.'),
    (r'\bcustom_sink_core\.', 'audio_sink_core.'),
    (r'\btranscription\.', 'transcription_service.'),
    (r'\btranscription_core\.', 'transcription_engine.'),
    (r'\btranslator\.', 'translation_service.'),
    (r'\btranslator_core\.', 'translation_engine.'),
    (r'\baudio_utils\.', 'audio_processing_utils.'),
    (r'\butils\.', 'translation_utils.'),
]

def backup_project():
    """Create a backup of the current project state."""
    backup_dir = Path('backup_before_rename')
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    print("📦 Creating backup...")
    shutil.copytree('.', backup_dir, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', 'logs', 'backup_*', '.git'
    ))
    print("✅ Backup created at backup_before_rename/")

def get_python_files():
    """Get all Python files in the project."""
    python_files = []
    for file_path in Path('.').glob('*.py'):
        if file_path.name not in ['rename_files.py', 'code_stats.py']:
            python_files.append(file_path)
    return python_files

def update_imports_in_file(file_path):
    """Update import statements in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Apply all import pattern replacements
        for pattern, replacement in IMPORT_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"  - {pattern} → {replacement}")
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📝 Updated imports in {file_path.name}:")
            for change in changes_made:
                print(change)
        
        return len(changes_made)
    
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return 0

def rename_files():
    """Rename files according to the rename map."""
    renamed_files = []
    
    for old_name, new_name in RENAME_MAP.items():
        old_path = Path(old_name)
        new_path = Path(new_name)
        
        if old_path.exists():
            print(f"🔄 Renaming {old_name} → {new_name}")
            old_path.rename(new_path)
            renamed_files.append((old_name, new_name))
        else:
            print(f"⚠️  File not found: {old_name}")
    
    return renamed_files

def update_all_imports():
    """Update imports in all Python files."""
    python_files = get_python_files()
    total_changes = 0
    
    print(f"\n📚 Updating imports in {len(python_files)} files...")
    
    for file_path in python_files:
        changes = update_imports_in_file(file_path)
        total_changes += changes
    
    print(f"\n✅ Updated imports in {total_changes} locations")

def verify_syntax():
    """Verify that all Python files have valid syntax."""
    python_files = get_python_files()
    errors = []
    
    print(f"\n🔍 Verifying syntax of {len(python_files)} files...")
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, str(file_path), 'exec')
            print(f"✅ {file_path.name}")
        except SyntaxError as e:
            error_msg = f"{file_path.name}: Line {e.lineno}: {e.msg}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")
        except Exception as e:
            error_msg = f"{file_path.name}: {e}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")
    
    return errors

def main():
    """Main function to execute the renaming process."""
    print("🎹 Discord Bot Translator - File Renaming Script")
    print("=" * 60)
    
    # Step 1: Create backup
    backup_project()
    
    # Step 2: Rename files
    print(f"\n🔄 Renaming {len(RENAME_MAP)} files...")
    renamed_files = rename_files()
    
    if not renamed_files:
        print("⚠️  No files were renamed. Exiting.")
        return
    
    # Step 3: Update imports
    update_all_imports()
    
    # Step 4: Verify syntax
    errors = verify_syntax()
    
    # Step 5: Summary
    print("\n" + "=" * 60)
    print("📊 RENAMING SUMMARY")
    print("=" * 60)
    print(f"✅ Files renamed: {len(renamed_files)}")
    
    for old_name, new_name in renamed_files:
        print(f"   {old_name} → {new_name}")
    
    if errors:
        print(f"\n❌ Syntax errors found: {len(errors)}")
        for error in errors:
            print(f"   {error}")
        print("\n⚠️  Please fix syntax errors before testing!")
    else:
        print(f"\n✅ All files have valid syntax!")
        print("\n🎯 Next steps:")
        print("   1. Test the bot to ensure everything works")
        print("   2. If issues arise, restore from backup_before_rename/")
        print("   3. If successful, commit the changes to git")

if __name__ == "__main__":
    main()
