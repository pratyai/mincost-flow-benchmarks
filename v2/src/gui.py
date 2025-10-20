import sys
import os
import subprocess
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
        self.setWindowTitle("MinCostFlow Benchmarks GUI")
        self.setGeometry(100, 100, 700, 500)  # Adjusted height for two list widgets

        # --- Instance variables for remembering last directories ---
        # Set default spec directory if available
        default_spec_path = os.path.join(JULIA_PROJECT_PATH, DEFAULT_SPEC_DIR_RELATIVE)
        if os.path.isdir(default_spec_path):
            self.last_input_spec_dir = default_spec_path
        else:
            self.last_input_spec_dir = os.getcwd()

        self.last_config_dir = os.getcwd()

        self.init_ui()
        self._load_default_input_specs()  # Load default input specs after UI is initialized
        self._load_default_configs()  # Load default configs after UI is initialized

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Input Spec Files (List Widget)
        input_spec_layout = QVBoxLayout()
        input_spec_label = QLabel("Input Spec Files (-i):")
        input_spec_layout.addWidget(input_spec_label)

        self.input_spec_list_widget = QListWidget()
        input_spec_layout.addWidget(self.input_spec_list_widget)

        input_spec_buttons_layout = QHBoxLayout()
        add_input_spec_button = QPushButton("➕")
        add_input_spec_button.clicked.connect(self.add_input_spec_file_to_list)
        input_spec_buttons_layout.addWidget(add_input_spec_button)

        remove_input_spec_button = QPushButton("➖")
        remove_input_spec_button.clicked.connect(self.remove_selected_input_specs)
        input_spec_buttons_layout.addWidget(remove_input_spec_button)

        clear_all_input_specs_button = QPushButton("🗑️")
        clear_all_input_specs_button.clicked.connect(self.clear_all_input_specs)
        input_spec_buttons_layout.addWidget(clear_all_input_specs_button)

        input_spec_layout.addLayout(input_spec_buttons_layout)

        main_layout.addLayout(input_spec_layout)

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
        main_layout.addLayout(output_db_h_layout)

        # Solution Directory
        self.solution_dir_edit = self._create_input_row(
            "Solution Directory (-s):", self.browse_directory
        )
        main_layout.addLayout(self.solution_dir_edit)

        # Config Files (List Widget)
        config_layout = QVBoxLayout()
        config_label = QLabel("Config Files (--configs):")
        config_layout.addWidget(config_label)

        self.config_list_widget = QListWidget()
        config_layout.addWidget(self.config_list_widget)

        config_buttons_layout = QHBoxLayout()
        add_config_button = QPushButton("➕")
        add_config_button.clicked.connect(self.add_config_file_to_list)
        config_buttons_layout.addWidget(add_config_button)

        remove_config_button = QPushButton("➖")
        remove_config_button.clicked.connect(self.remove_selected_configs)
        config_buttons_layout.addWidget(remove_config_button)

        clear_all_configs_button = QPushButton("🗑️")
        clear_all_configs_button.clicked.connect(self.clear_all_configs)
        config_buttons_layout.addWidget(clear_all_configs_button)

        config_layout.addLayout(config_buttons_layout)

        main_layout.addLayout(config_layout)

        # Run and Cancel Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Benchmarks")
        self.run_button.clicked.connect(self.run_benchmarks)
        button_layout.addWidget(self.run_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _create_input_row(self, label_text, browse_func, button_text="Browse"):
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
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
            self, "Select Directory", initial_path
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
            # No need to update last_input_spec_dir here, it's already set in __init__

    def add_input_spec_file_to_list(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Add Input Spec File", self.last_input_spec_dir
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
            JULIA_PROJECT_PATH, DEFAULT_CONFIG_DIR_RELATIVE
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
            self.last_config_dir = (
                default_config_path  # Set last config dir to the default path
            )

    def add_config_file_to_list(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Add Config File", self.last_config_dir
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

    def run_benchmarks(self):
        julia_command = [
            "julia",
            "--project=.",
            os.path.join(JULIA_PROJECT_PATH, "src", "main.jl"),
        ]

        # Get input spec files from QListWidget
        input_specs = [
            self.input_spec_list_widget.item(i).text()
            for i in range(self.input_spec_list_widget.count())
        ]

        output_db = (
            self.output_db_line_edit.text()
        )  # Get text from the specific QLineEdit
        solution_dir = self.solution_dir_edit.itemAt(1).widget().text()

        # Get config files from QListWidget
        configs = [
            self.config_list_widget.item(i).text()
            for i in range(self.config_list_widget.count())
        ]

        if input_specs:
            for input_spec_file in input_specs:
                if input_spec_file.strip():
                    julia_command.extend(["-i", input_spec_file.strip()])
        # Only add -o if the field is not empty
        if output_db:
            julia_command.extend(["-o", output_db])
        if solution_dir:
            julia_command.extend(["-s", solution_dir])
        if configs:
            for config_file in configs:
                if config_file.strip():
                    julia_command.extend(["--configs", config_file.strip()])

        print("Executing Julia command:", " ".join(julia_command))
        try:
            result = subprocess.run(
                julia_command,
                capture_output=True,
                text=True,
                check=True,
                cwd=JULIA_PROJECT_PATH,
            )
            QMessageBox.information(
                self,
                "Success",
                "Julia command executed successfully!\n\nStdout:\n"
                + result.stdout
                + "\nStderr:\n"
                + result.stderr,
            )
            print("Julia Stdout:\n", result.stdout)
            print("Julia Stderr:\n", result.stderr)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self,
                "Error",
                "Julia command failed!\n\nReturn Code: "
                + str(e.returncode)
                + "\nStdout:\n"
                + e.stdout
                + "\nStderr:\n"
                + e.stderr,
            )
            print("Julia command failed!")
            print("Return Code:", e.returncode)
            print("Stdout:\n", e.stdout)
            print("Stderr:\n", e.stderr)
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "Error",
                "'julia' command not found. Is Julia installed and in your PATH?",
            )
            print(
                "Error: 'julia' command not found. Is Julia installed and in your PATH?"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MinCostFlowGUI()
    gui.show()
    sys.exit(app.exec_())
