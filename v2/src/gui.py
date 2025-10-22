import sys
import os
import subprocess
import csv
import tempfile
import shutil
import json
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QStackedWidget,
    QSplitter,
)
from PyQt5.QtCore import Qt

# --- Configuration ---
JULIA_PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Path to the Julia project root
DEFAULT_SPEC_DIR_RELATIVE = os.path.join(
    "data", "specs"
)  # Relative path to default spec directory
DEFAULT_CONFIG_DIR_RELATIVE = os.path.join(
    "configs"
)  # Relative path to default config directory


class MinCostFlowGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.run_command_on_exit = False
        self.setWindowTitle("MinCostFlow Benchmarks GUI")
        self.setGeometry(100, 100, 900, 600)  # Adjusted size for side panel

        self.last_run_file = ".last_run.json"

        # --- Instance variables for remembering last directories ---
        default_spec_path = os.path.join(JULIA_PROJECT_PATH, DEFAULT_SPEC_DIR_RELATIVE)
        self.last_input_spec_dir = (
            default_spec_path if os.path.isdir(default_spec_path) else os.getcwd()
        )
        self.last_config_dir = os.getcwd()

        self.problem_selections = {}

        self.init_ui()
        self._load_default_input_specs()
        self._load_default_configs()

        if os.path.exists(self.last_run_file):
            self.load_last_run_button.setEnabled(True)

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left Panel (Main Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Input Spec Files
        input_spec_layout = self._create_list_widget_layout(
            "Input Spec Files (-i):",
            self.add_input_spec_file_to_list,
            self.remove_selected_input_specs,
            self.clear_all_input_specs,
        )
        self.input_spec_list_widget = input_spec_layout.itemAt(1).widget()
        left_layout.addLayout(input_spec_layout)

        # Config Files
        config_layout = self._create_list_widget_layout(
            "Config Files (-c):",
            self.add_config_file_to_list,
            self.remove_selected_configs,
            self.clear_all_configs,
        )
        self.config_list_widget = config_layout.itemAt(1).widget()
        left_layout.addLayout(config_layout)

        # Output Database Path
        output_db_h_layout = QHBoxLayout()
        output_db_label = QLabel("Output Database Path (-o):")
        self.output_db_line_edit = QLineEdit()
        self.output_db_line_edit.setPlaceholderText(
            "Leave empty to use spec_name.db for each spec"
        )
        output_db_browse_button = QPushButton("Browse")
        output_db_browse_button.clicked.connect(
            lambda: self.browse_file(self.output_db_line_edit)
        )
        output_db_h_layout.addWidget(output_db_label)
        output_db_h_layout.addWidget(self.output_db_line_edit)
        output_db_h_layout.addWidget(output_db_browse_button)
        left_layout.addLayout(output_db_h_layout)

        # Solution Directory
        self.solution_dir_edit = self._create_input_row(
            "Solution Directory (-s):",
            self.browse_directory,
            placeholder_text="Leave empty to not generate solution files",
        )
        left_layout.addLayout(self.solution_dir_edit)

        # Run and Cancel Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Benchmarks")
        self.run_button.clicked.connect(self.run_benchmarks)
        button_layout.addWidget(self.run_button)

        self.load_last_run_button = QPushButton("Load Last Run")
        self.load_last_run_button.setEnabled(False)
        self.load_last_run_button.clicked.connect(self.load_last_run)
        button_layout.addWidget(self.load_last_run_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        button_layout.addWidget(self.cancel_button)
        left_layout.addLayout(button_layout)

        # --- Right Panel (Dynamic Viewer) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.side_panel_label = QLabel("Details")
        right_layout.addWidget(self.side_panel_label)

        self.stacked_widget = QStackedWidget()
        right_layout.addWidget(self.stacked_widget)

        # Problems List (for spec files)
        self.problems_list_widget = QListWidget()
        self.stacked_widget.addWidget(self.problems_list_widget)

        # Config Content Viewer
        self.config_content_viewer = QTextEdit()
        self.config_content_viewer.setReadOnly(True)
        self.stacked_widget.addWidget(self.config_content_viewer)

        # --- Add panels to splitter ---
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 300])  # Initial sizes
        main_layout.addWidget(splitter)

        # --- Connect signals ---
        self.input_spec_list_widget.itemSelectionChanged.connect(
            self.on_spec_selection_changed
        )
        self.config_list_widget.itemSelectionChanged.connect(
            self.on_config_selection_changed
        )
        self.problems_list_widget.itemChanged.connect(self.on_problem_selection_changed)

    def _create_list_widget_layout(self, label_text, add_func, remove_func, clear_func):
        layout = QVBoxLayout()
        label = QLabel(label_text)
        layout.addWidget(label)

        list_widget = QListWidget()
        layout.addWidget(list_widget)

        buttons_layout = QHBoxLayout()
        add_button = QPushButton("➕")
        add_button.clicked.connect(add_func)
        buttons_layout.addWidget(add_button)

        remove_button = QPushButton("➖")
        remove_button.clicked.connect(remove_func)
        buttons_layout.addWidget(remove_button)

        clear_button = QPushButton("🗑️")
        clear_button.clicked.connect(clear_func)
        buttons_layout.addWidget(clear_button)

        layout.addLayout(buttons_layout)
        return layout

    def on_spec_selection_changed(self):
        if self.input_spec_list_widget.selectedItems():
            self.config_list_widget.clearSelection()
            self.side_panel_label.setText("Problems in selected spec file:")
            self.stacked_widget.setCurrentWidget(self.problems_list_widget)
            self.update_problems_list()

    def on_config_selection_changed(self):
        if self.config_list_widget.selectedItems():
            self.input_spec_list_widget.clearSelection()
            self.side_panel_label.setText("Config File Content:")
            self.stacked_widget.setCurrentWidget(self.config_content_viewer)
            self.update_config_view()

    def update_config_view(self):
        selected_items = self.config_list_widget.selectedItems()
        if not selected_items:
            self.config_content_viewer.clear()
            return

        config_file = selected_items[0].text()
        try:
            with open(config_file, "r") as f:
                self.config_content_viewer.setText(f.read())
        except Exception as e:
            self.config_content_viewer.setText(f"Error reading file: {e}")

    def _create_input_row(
        self,
        label_text,
        browse_func,
        button_text="Browse",
        placeholder_text=None,
    ):
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        if placeholder_text:
            line_edit.setPlaceholderText(placeholder_text)
        browse_button = QPushButton(button_text)
        browse_button.clicked.connect(lambda: browse_func(line_edit))

        h_layout.addWidget(label)
        h_layout.addWidget(line_edit)
        h_layout.addWidget(browse_button)
        return h_layout

    def browse_file(self, line_edit):
        initial_path = (
            os.path.dirname(line_edit.text()) if line_edit.text() else os.getcwd()
        )
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", initial_path)
        if file_path:
            line_edit.setText(file_path)

    def browse_directory(self, line_edit):
        initial_path = line_edit.text() if line_edit.text() else os.getcwd()
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            initial_path,
        )
        if dir_path:
            line_edit.setText(dir_path)

    def _load_default_input_specs(self):
        default_spec_path = os.path.join(JULIA_PROJECT_PATH, DEFAULT_SPEC_DIR_RELATIVE)
        warmup_spec_path = os.path.join(default_spec_path, "warmup.inspec")
        if os.path.isfile(warmup_spec_path):
            if warmup_spec_path not in [
                self.input_spec_list_widget.item(i).text()
                for i in range(self.input_spec_list_widget.count())
            ]:
                self.input_spec_list_widget.addItem(warmup_spec_path)

    def add_input_spec_file_to_list(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Add Input Spec File",
            self.last_input_spec_dir,
        )
        if file_path:
            if file_path not in [
                self.input_spec_list_widget.item(i).text()
                for i in range(self.input_spec_list_widget.count())
            ]:
                self.input_spec_list_widget.addItem(file_path)
            self.last_input_spec_dir = os.path.dirname(file_path)

    def remove_selected_input_specs(self):
        for item in self.input_spec_list_widget.selectedItems():
            self.input_spec_list_widget.takeItem(self.input_spec_list_widget.row(item))

    def clear_all_input_specs(self):
        self.input_spec_list_widget.clear()

    def _load_default_configs(self):
        default_config_path = os.path.join(
            JULIA_PROJECT_PATH,
            DEFAULT_CONFIG_DIR_RELATIVE,
        )
        if os.path.isdir(default_config_path):
            for root, _, files in os.walk(default_config_path):
                for file in files:
                    if file.endswith(".toml"):
                        full_path = os.path.join(root, file)
                        if full_path not in [
                            self.config_list_widget.item(i).text()
                            for i in range(self.config_list_widget.count())
                        ]:
                            self.config_list_widget.addItem(full_path)
            self.last_config_dir = default_config_path

    def add_config_file_to_list(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Add Config File",
            self.last_config_dir,
        )
        if file_path:
            if file_path not in [
                self.config_list_widget.item(i).text()
                for i in range(self.config_list_widget.count())
            ]:
                self.config_list_widget.addItem(file_path)
            self.last_config_dir = os.path.dirname(file_path)

    def remove_selected_configs(self):
        for item in self.config_list_widget.selectedItems():
            self.config_list_widget.takeItem(self.config_list_widget.row(item))

    def clear_all_configs(self):
        self.config_list_widget.clear()

    def update_problems_list(self):
        selected_items = self.input_spec_list_widget.selectedItems()

        try:
            self.problems_list_widget.itemChanged.disconnect(
                self.on_problem_selection_changed
            )
        except TypeError:  # Was not connected
            pass

        self.problems_list_widget.clear()

        if not selected_items:
            return

        selected_spec_file = selected_items[0].text()

        try:
            with open(selected_spec_file, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)

                for i, row in enumerate(reader):
                    if not row:
                        continue
                    problem_name = row[0]
                    item = QListWidgetItem(problem_name)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

                    if selected_spec_file in self.problem_selections:
                        if i in self.problem_selections[selected_spec_file]:
                            item.setCheckState(Qt.Checked)
                        else:
                            item.setCheckState(Qt.Unchecked)
                    else:
                        item.setCheckState(Qt.Checked)

                    self.problems_list_widget.addItem(item)
        except Exception as e:
            print(f"Error reading spec file: {e}")
        finally:
            self.problems_list_widget.itemChanged.connect(
                self.on_problem_selection_changed
            )

    def on_problem_selection_changed(self, item):
        selected_spec_items = self.input_spec_list_widget.selectedItems()
        if not selected_spec_items:
            return

        selected_spec_file = selected_spec_items[0].text()

        if selected_spec_file not in self.problem_selections:
            all_rows = set(range(self.problems_list_widget.count()))
            self.problem_selections[selected_spec_file] = all_rows

        row = self.problems_list_widget.row(item)
        if item.checkState() == Qt.Checked:
            self.problem_selections[selected_spec_file].add(row)
        else:
            self.problem_selections[selected_spec_file].discard(row)

    def save_last_run(self):
        problem_selections_serializable = {
            k: list(v) for k, v in self.problem_selections.items()
        }
        data = {
            "input_specs": [
                self.input_spec_list_widget.item(i).text()
                for i in range(self.input_spec_list_widget.count())
            ],
            "configs": [
                self.config_list_widget.item(i).text()
                for i in range(self.config_list_widget.count())
            ],
            "output_db": self.output_db_line_edit.text(),
            "solution_dir": self.solution_dir_edit.itemAt(1).widget().text(),
            "problem_selections": problem_selections_serializable,
        }
        try:
            with open(self.last_run_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving last run configuration: {e}")

    def load_last_run(self):
        try:
            with open(self.last_run_file, "r") as f:
                data = json.load(f)

            self.input_spec_list_widget.clear()
            self.input_spec_list_widget.addItems(data.get("input_specs", []))

            self.config_list_widget.clear()
            self.config_list_widget.addItems(data.get("configs", []))

            self.output_db_line_edit.setText(data.get("output_db", ""))
            self.solution_dir_edit.itemAt(1).widget().setText(
                data.get("solution_dir", "")
            )

            self.problem_selections = {
                k: set(v) for k, v in data.get("problem_selections", {}).items()
            }

            # Refresh views
            if self.input_spec_list_widget.count() > 0:
                self.input_spec_list_widget.setCurrentRow(0)
                self.on_spec_selection_changed()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load last run configuration: {e}"
            )

    def run_benchmarks(self):
        self.save_last_run()

        julia_command = [
            "julia",
            "--project=.",
            os.path.join(JULIA_PROJECT_PATH, "src", "main.jl"),
        ]

        input_specs = [
            self.input_spec_list_widget.item(i).text()
            for i in range(self.input_spec_list_widget.count())
        ]

        processed_input_specs = []
        temp_dir = tempfile.mkdtemp()
        self.temp_dir_to_cleanup = temp_dir

        for spec_file in input_specs:
            if spec_file in self.problem_selections:
                try:
                    with open(spec_file, "r", newline="") as f_in:
                        reader = csv.reader(f_in)
                        header = next(reader)
                        all_rows = list(reader)

                    selected_rows_indices = self.problem_selections[spec_file]

                    base_name = os.path.basename(spec_file)
                    temp_path = os.path.join(temp_dir, base_name)

                    with open(temp_path, "w", newline="") as f_out:
                        writer = csv.writer(f_out)
                        writer.writerow(header)
                        for i, row in enumerate(all_rows):
                            if i in selected_rows_indices:
                                writer.writerow(row)

                    processed_input_specs.append(temp_path)

                except Exception as e:
                    print(f"Error processing spec file {spec_file}: {e}")
                    processed_input_specs.append(spec_file)
            else:
                processed_input_specs.append(spec_file)

        output_db = self.output_db_line_edit.text()
        solution_dir = self.solution_dir_edit.itemAt(1).widget().text()

        configs = [
            self.config_list_widget.item(i).text()
            for i in range(self.config_list_widget.count())
        ]

        if processed_input_specs:
            for input_spec_file in processed_input_specs:
                if input_spec_file.strip():
                    julia_command.extend(["-i", input_spec_file.strip()])
        if output_db:
            julia_command.extend(["-o", output_db])
        if solution_dir:
            julia_command.extend(["-s", solution_dir])
        if configs:
            for config_file in configs:
                if config_file.strip():
                    julia_command.extend(["-c", config_file.strip()])

        self.run_command_on_exit = True
        self.julia_command = julia_command
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MinCostFlowGUI()
    gui.show()
    app.exec_()

    if not getattr(gui, "run_command_on_exit", False):
        sys.exit(0)

    julia_command = gui.julia_command
    return_code = 0

    print("Executing Julia command:", " ".join(julia_command))
    try:
        process = subprocess.Popen(
            julia_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=JULIA_PROJECT_PATH,
            bufsize=1,
            env=dict(os.environ, JULIA_NUM_THREADS="2"),
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
            process.stdout.close()

        process.wait()
        return_code = process.returncode

        if return_code == 0:
            print("\nJulia command executed successfully!")
        else:
            print(f"\nJulia command failed with return code: {return_code}")

    except FileNotFoundError:
        print("Error: 'julia' command not found. Is Julia installed and in your PATH?")
        return_code = 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return_code = 1
    finally:
        if hasattr(gui, "temp_dir_to_cleanup"):
            try:
                shutil.rmtree(gui.temp_dir_to_cleanup)
                print(f"Cleaned up temp directory: {gui.temp_dir_to_cleanup}")
            except OSError as e:
                print(
                    f"Error cleaning up temp directory {gui.temp_dir_to_cleanup}: {e}"
                )

    sys.exit(return_code)
