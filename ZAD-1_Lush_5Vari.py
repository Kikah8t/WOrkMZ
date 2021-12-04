import numpy as np
from matplotlib import pyplot as plt
import os
import xml.etree.ElementTree as ET
import shutil
import xml.dom.minidom

#функция создания xml
def create_xml(x, y):
    data = ET.Element("data")
    xdata = ET.SubElement(data, "xdata")
    for num in range(len(x)):
        x_name = ET.SubElement(xdata, "x")
        x_name.text = str(x[num])

    ydata = ET.SubElement(data, "ydata")
    for num in range(len(y)):
        y_name = ET.SubElement(ydata, "y")
        y_name.text = str(y[num])

    tree = ET.ElementTree(data)
    tree.write("result.xml", encoding='UTF-8', xml_declaration=True)

#Создание директивы
path = os.getcwd()
shutil.rmtree(path + "/result", ignore_errors=True)
os.mkdir(path + "/result")
os.chdir(path + "/result")

#Задание функций
dx = 0.01
x = np.arange(-100.0, 101.0, dx)
y = -np.cos(x)*np.cos(np.pi)*np.exp(-(x-np.pi)*(x-np.pi))

#Создание файла и его редактирование
create_xml(x, y)
parent_file_path = 'result.xml'
parent_tree = ET.parse(parent_file_path)
parent = parent_tree.getroot()
xmlstr = xml.dom.minidom.parseString(ET.tostring(parent)).toprettyxml()
f = open(parent_file_path, "w")
f.write(xmlstr)
f.close()

#Построение функции
plt.subplots()
plt.title("График заданной функции")
plt.ylabel('Ось y')
plt.xlabel('Ось x')
plt.axis([-1, 5, -2, 0.1])
plt.grid()
plt.plot(x,y)
plt.show()


