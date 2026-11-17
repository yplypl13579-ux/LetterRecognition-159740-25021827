QT += core gui widgets
CONFIG += c++17 release
CONFIG -= debug debug_and_release
TARGET = LetterRecognition
TEMPLATE = app
SOURCES += main.cpp mainwindow.cpp backpropagation.cpp
HEADERS += mainwindow.h backpropagation.h default_config.h
