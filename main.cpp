#include "mainwindow.h"
#include <QApplication>
#include <QDir>
#include <QFontDatabase>
#include <QTimer>
#include <iostream>
int main(int argc, char **argv) {
  QApplication app(argc, argv);
  if (app.arguments().contains("--smoke-test")) {
    QFontDatabase::addApplicationFont(qEnvironmentVariable("WINDIR", "C:/Windows") +
                                      "/Fonts/arial.ttf");
  }
  app.setFont(QFont("Arial", 10));
  MainWindow w;
  w.show();
  if (app.arguments().contains("--smoke-test")) {
    QString scratch = app.arguments().value(app.arguments().indexOf("--smoke-test") + 1);
    QTimer::singleShot(0, &w, [&, scratch] {
      try {
        w.smokeTest(scratch);
        app.processEvents();
        if (!w.grab().save(QDir(scratch).filePath("gui_smoke.png")))
          throw std::runtime_error("Screenshot save failed");
        std::cout << "GUI SMOKE TEST PASSED\n";
        app.exit(0);
      } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        app.exit(1);
      }
    });
  }
  return app.exec();
}
