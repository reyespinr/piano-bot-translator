"""
Code Statistics Analyzer for Discord Bot Translator Project.

This script analyzes the Python codebase to provide insights into:
- Total lines of code across all .py files
- Code distribution per file
- Comments vs code ratio
- Documentation coverage
- Progress tracking over time

Run this script to get a comprehensive overview of your project's size and complexity.
"""
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class FileStats:
    """Statistics for a single Python file."""
    filename: str
    total_lines: int
    code_lines: int
    comment_lines: int
    docstring_lines: int
    blank_lines: int
    functions: int
    classes: int
    imports: int
    size_bytes: int


@dataclass
class ProjectStats:
    """Overall project statistics."""
    total_files: int
    total_lines: int
    total_code_lines: int
    total_comment_lines: int
    total_docstring_lines: int
    total_blank_lines: int
    total_functions: int
    total_classes: int
    total_imports: int
    total_size_bytes: int
    files: List[FileStats]
    analysis_date: str


class CodeAnalyzer:
    """Analyzes Python code statistics."""

    def __init__(self, project_path: str = "."):
        """Initialize the analyzer with project path."""
        self.project_path = Path(project_path)
        self.stats_history_file = self.project_path / "code_stats_history.json"

    def analyze_file(self, file_path: Path) -> FileStats:
        """Analyze a single Python file and return statistics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
        except (UnicodeDecodeError, IOError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return FileStats(
                filename=file_path.name,
                total_lines=0, code_lines=0, comment_lines=0,
                docstring_lines=0, blank_lines=0, functions=0,
                classes=0, imports=0, size_bytes=0
            )

        stats = FileStats(
            filename=file_path.name,
            total_lines=len(lines),
            code_lines=0,
            comment_lines=0,
            docstring_lines=0,
            blank_lines=0,
            functions=0,
            classes=0,
            imports=0,
            size_bytes=file_path.stat().st_size
        )

        in_multiline_string = False
        multiline_delimiter = None

        for line in lines:
            stripped = line.strip()

            # Count blank lines
            if not stripped:
                stats.blank_lines += 1
                continue

            # Handle multiline strings (docstrings)
            if not in_multiline_string:
                # Check for start of multiline string
                if '"""' in stripped or "'''" in stripped:
                    if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                        # Single line triple quote
                        stats.docstring_lines += 1
                    else:
                        # Start of multiline
                        in_multiline_string = True
                        multiline_delimiter = '"""' if '"""' in stripped else "'''"
                        stats.docstring_lines += 1
                    continue
            else:
                # We're in a multiline string
                stats.docstring_lines += 1
                if multiline_delimiter in stripped:
                    in_multiline_string = False
                    multiline_delimiter = None
                continue

            # Count comments
            if stripped.startswith('#'):
                stats.comment_lines += 1
                continue

            # Count code constructs
            if re.match(r'^def\s+\w+\s*\(', stripped):
                stats.functions += 1
            elif re.match(r'^class\s+\w+', stripped):
                stats.classes += 1
            elif re.match(r'^(import\s+|from\s+\w+.*import)', stripped):
                stats.imports += 1

            # If we get here, it's a code line
            stats.code_lines += 1

        return stats

    def analyze_project(self) -> ProjectStats:
        """Analyze all Python files in the project."""
        python_files = list(self.project_path.glob("*.py"))
        file_stats = []

        print(f"Analyzing {len(python_files)} Python files...")
        print("=" * 60)

        for file_path in sorted(python_files):
            stats = self.analyze_file(file_path)
            file_stats.append(stats)

            # Print individual file stats
            print(f"{stats.filename:<25} | "
                  f"Lines: {stats.total_lines:>4} | "
                  f"Code: {stats.code_lines:>4} | "
                  f"Comments: {stats.comment_lines:>3} | "
                  f"Docs: {stats.docstring_lines:>3} | "
                  f"Functions: {stats.functions:>2} | "
                  f"Classes: {stats.classes:>2}")

        # Calculate totals
        project_stats = ProjectStats(
            total_files=len(file_stats),
            total_lines=sum(f.total_lines for f in file_stats),
            total_code_lines=sum(f.code_lines for f in file_stats),
            total_comment_lines=sum(f.comment_lines for f in file_stats),
            total_docstring_lines=sum(f.docstring_lines for f in file_stats),
            total_blank_lines=sum(f.blank_lines for f in file_stats),
            total_functions=sum(f.functions for f in file_stats),
            total_classes=sum(f.classes for f in file_stats),
            total_imports=sum(f.imports for f in file_stats),
            total_size_bytes=sum(f.size_bytes for f in file_stats),
            files=file_stats,
            analysis_date=datetime.now().isoformat()
        )

        return project_stats

    def print_summary(self, stats: ProjectStats):
        """Print a comprehensive summary of the project statistics."""
        print("\n" + "=" * 60)
        print("🎹 DISCORD BOT TRANSLATOR - CODE STATISTICS")
        print("=" * 60)

        print("📊 PROJECT OVERVIEW:")
        print(f"   • Total Python files: {stats.total_files}")
        print(
            f"   • Total project size: {stats.total_size_bytes / 1024:.1f} KB")
        print(
            f"   • Analysis date: {datetime.fromisoformat(stats.analysis_date).strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n📝 LINE BREAKDOWN:")
        print(f"   • Total lines: {stats.total_lines:,}")
        print(
            f"   • Code lines: {stats.total_code_lines:,} ({stats.total_code_lines/stats.total_lines*100:.1f}%)")
        print(
            f"   • Comment lines: {stats.total_comment_lines:,} ({stats.total_comment_lines/stats.total_lines*100:.1f}%)")
        print(
            f"   • Documentation lines: {stats.total_docstring_lines:,} ({stats.total_docstring_lines/stats.total_lines*100:.1f}%)")
        print(
            f"   • Blank lines: {stats.total_blank_lines:,} ({stats.total_blank_lines/stats.total_lines*100:.1f}%)")

        print("\n🏗️ CODE STRUCTURE:")
        print(f"   • Functions: {stats.total_functions}")
        print(f"   • Classes: {stats.total_classes}")
        print(f"   • Import statements: {stats.total_imports}")

        # Calculate some derived metrics
        avg_lines_per_file = stats.total_lines / \
            stats.total_files if stats.total_files > 0 else 0
        avg_functions_per_file = stats.total_functions / \
            stats.total_files if stats.total_files > 0 else 0
        documentation_ratio = (stats.total_comment_lines + stats.total_docstring_lines) / \
            stats.total_code_lines if stats.total_code_lines > 0 else 0

        print("\n📈 QUALITY METRICS:")
        print(f"   • Average lines per file: {avg_lines_per_file:.1f}")
        print(f"   • Average functions per file: {avg_functions_per_file:.1f}")
        print(
            f"   • Documentation ratio: {documentation_ratio:.2f} (comments+docs per code line)")

        # Show largest files
        largest_files = sorted(
            stats.files, key=lambda x: x.total_lines, reverse=True)[:5]
        print("\n📋 LARGEST FILES:")
        for i, file_stat in enumerate(largest_files, 1):
            print(
                f"   {i}. {file_stat.filename}: {file_stat.total_lines} lines ({file_stat.code_lines} code)")

        # Show most complex files (by function count)
        complex_files = sorted(
            stats.files, key=lambda x: x.functions, reverse=True)[:5]
        print("\n🔧 MOST COMPLEX FILES (by function count):")
        for i, file_stat in enumerate(complex_files, 1):
            print(
                f"   {i}. {file_stat.filename}: {file_stat.functions} functions")

    def save_history(self, stats: ProjectStats):
        """Save statistics to history file for progress tracking."""
        try:
            # Load existing history
            history = []
            if self.stats_history_file.exists():
                with open(self.stats_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

            # Convert stats to dict for JSON serialization
            stats_dict = {
                'analysis_date': stats.analysis_date,
                'total_files': stats.total_files,
                'total_lines': stats.total_lines,
                'total_code_lines': stats.total_code_lines,
                'total_comment_lines': stats.total_comment_lines,
                'total_docstring_lines': stats.total_docstring_lines,
                'total_blank_lines': stats.total_blank_lines,
                'total_functions': stats.total_functions,
                'total_classes': stats.total_classes,
                'total_imports': stats.total_imports,
                'total_size_bytes': stats.total_size_bytes
            }

            # Add to history
            history.append(stats_dict)

            # Keep only last 50 entries
            history = history[-50:]

            # Save back to file
            with open(self.stats_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

            print(f"\n💾 Statistics saved to {self.stats_history_file}")

        except Exception as e:
            print(f"Warning: Could not save statistics history: {e}")

    def show_progress(self):
        """Show progress over time if history exists."""
        if not self.stats_history_file.exists():
            print(
                "\n📊 No history available yet. Run this script a few times to see progress!")
            return

        try:
            with open(self.stats_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            if len(history) < 2:
                print("\n📊 Need at least 2 analyses to show progress.")
                return

            # Compare latest with previous
            latest = history[-1]
            previous = history[-2]

            print("\n📈 PROGRESS SINCE LAST ANALYSIS:")
            print(
                f"   • Lines: {latest['total_lines'] - previous['total_lines']:+d}")
            print(
                f"   • Code lines: {latest['total_code_lines'] - previous['total_code_lines']:+d}")
            print(
                f"   • Functions: {latest['total_functions'] - previous['total_functions']:+d}")
            print(
                f"   • Classes: {latest['total_classes'] - previous['total_classes']:+d}")

            # Show trend over last 5 analyses
            if len(history) >= 5:
                recent = history[-5:]
                first = recent[0]
                last = recent[-1]

                print("\n📊 TREND (last 5 analyses):")
                print(
                    f"   • Total growth: {last['total_lines'] - first['total_lines']:+d} lines")
                print(
                    f"   • Average per analysis: {(last['total_lines'] - first['total_lines']) / 4:.1f} lines")

        except Exception as e:
            print(f"Warning: Could not load progress history: {e}")


def main():
    """Main function to run the code analysis."""
    print("🎹 Discord Bot Translator - Code Statistics Analyzer")
    print("Starting analysis...\n")

    analyzer = CodeAnalyzer()
    stats = analyzer.analyze_project()

    analyzer.print_summary(stats)
    analyzer.show_progress()
    analyzer.save_history(stats)

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("\nTip: Run this script regularly to track your refactoring progress!")


if __name__ == "__main__":
    main()
