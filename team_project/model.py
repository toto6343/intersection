from roboflow import Roboflow
rf = Roboflow(api_key="EOcgTkCLUc6sFR8Pv6Lf")
project = rf.workspace("joyk-cl8nt").project("project-twhf4")
version = project.version(1)
dataset = version.download("yolov11")
                