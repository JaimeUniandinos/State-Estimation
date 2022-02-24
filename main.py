import sys

sys.path.append(r"E:\Program Files\DIgSILENT\PowerFactory 2020 SP2A\Python\3.7")
# import  PowerFactory  module
import powerfactory
# start PowerFactory  in engine  mode
app = powerfactory.GetApplication()

user = app.GetCurrentUser()

# activate project
project = app.ActivateProject("Nine Bus System")
prj = app.GetActiveProject()
print(prj)