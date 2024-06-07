import sys
import re
import spacy
import joblib
import pickle
import sklearn
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QPushButton, QLabel, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.Qt import QHeaderView
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


class EventTable(QTableWidget):
    def __init__(self, data, details_text, *args):
        QTableWidget.__init__(self, *args)
        self.data = data
        self.details_text = details_text
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setData()
        self.cellClicked.connect(self.showEventDetails)

    def setData(self):
        self.setRowCount(len(self.data))
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Дата", "Тип события"])

        max_type_width = 10
        max_date_width = 10
        for row, (date, event_type, _) in enumerate(self.data):
            formatted_date = datetime.utcfromtimestamp(float(date)).strftime('%Y-%m-%d %H:%M:%S')

            date_item = QTableWidgetItem(formatted_date)
            date_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.setItem(row, 0, date_item)

            type_item = QTableWidgetItem(str(event_type))
            type_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.setItem(row, 1, type_item)

            max_type_width = max(max_type_width, QFontMetrics(self.font()).width(str(event_type)))
            max_date_width = max(max_date_width, QFontMetrics(self.font()).width(formatted_date))

        self.setColumnWidth(0, max_date_width + 20)
        self.setColumnWidth(1, max_type_width + 20)

    def showEventDetails(self, row, column):
        event_info = f"Дата: {self.item(row, 0).text()}\nТип события: {self.item(row, 1).text()}\n\nДополнительная информация:\n"
        doc = nlp(self.data[row][2])
        log_details = [(ent.text, ent.label_) for ent in doc.ents]
        for param, label in log_details:
            value_after_equal = param.split('=', 1)[1]
            event_info += f"{label}: {value_after_equal}\n"
        self.details_text.setText(event_info)

class LogProcessor(QThread):
    dataReady = pyqtSignal(list)

    def __init__(self, file_name, svm_model, tfidf_vectorizer, parent=None):
        super().__init__(parent)
        self.file_name = file_name
        self.svm_model = svm_model
        self.tfidf_vectorizer = tfidf_vectorizer

    def run(self):
        data = []
        try:
            with open(self.file_name, 'r') as file:
                for line in file:
                    match = re.match(r'^type=(?P<type>\w+).*?msg=audit\((?P<date>\d+\.\d+):\d+\):\s*(?P<info>.*)$', line)
                    if match:
                        event_date = match.group('date')
                        event_type = self.svm_model.predict(self.tfidf_vectorizer.transform([line]))[0]
                        event_info = match.group('info')
                        data.append((event_date, event_type, event_info))
        except Exception as e:
            print(f"Error processing file: {e}")

        self.dataReady.emit(data)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Analyzer")
        self.setGeometry(100, 100, 700, 600)
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.load_button = QPushButton('Загрузить файл с логами')
        self.load_button.clicked.connect(self.loadFile)
        left_layout.addWidget(self.load_button)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)

        self.table_widget = EventTable([], self.details_text, 0, 2)
        left_layout.addWidget(self.table_widget)

        left_container = QWidget()
        left_container.setLayout(left_layout)

        right_layout = QVBoxLayout()
        self.details_label = QLabel("Инфо")

        right_layout.addWidget(self.details_label)
        right_layout.addWidget(self.details_text)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        layout.addWidget(left_container, 75)
        layout.addWidget(right_container, 25)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    def loadFile(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с логами", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            self.processFile(file_name)

    def processFile(self, file_name):
        self.thread = LogProcessor(file_name, loaded_svm_model, loaded_tfidf_vectorizer)
        self.thread.dataReady.connect(self.updateTable)
        self.thread.start()

    def updateTable(self, data):
        self.table_widget.data = data
        self.table_widget.setData()


if __name__ == "__main__":
    loaded_svm_model = joblib.load('type_pred_model/svm_model.joblib')
    loaded_tfidf_vectorizer = joblib.load('type_pred_model/tfidf_vectorizer.joblib')
    nlp = spacy.load("parse_model")

    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())