# URDF kitchen Tutorial  
[English](Tutorial_EN.md) | [日本語](Tutorial_JP.md)  
  
Let's build URDF/MJCF robot models with a GUI.  
  
---  
## Overview  
  
**URDF kitchen** is a set of tools that lets you **assemble robot structures in a GUI from mesh assets such as STL/OBJ/DAE, and export them as URDF/MJCF**.  
Like following a recipe, URDF kitchen breaks down the creation of URDF/MJCF robot models into clear, intuitive steps.  
  
## Target Users  
  
The basic prerequisite is that you already have a robot model created in a CAD or 3D modeling application.  
This tool is useful when your software cannot directly export URDF/MJCF, or when auto-generated output needs manual adjustments.  
It is especially valuable for anyone who has been writing URDF/MJCF by hand in a text editor.  
  
---  
  
## Installation  
  
Tested on Python 3.11; may work on newer versions.  
Set up a Python virtual environment and install the following libraries:  
  
```  
pip install numpy PySide6 vtk NodeGraphQt trimesh pycollada networkx xacrodoc
```  
  
Navigate to the directory containing the six files (urdf_kitchen_Launcher.py, urdf_kitchen_MeshSourcer.py, urdf_kitchen_PartsEditor.py, urdf_kitchen_Assembler.py, urdf_kitchen_Importer.py, urdf_kitchen_utils.py) and run:  
  
```  
python urdf_kitchen_Launcher.py  
```  
  
The launcher will start, giving you access to Mesh Sourcer, Parts Editor, and Assembler.  
  
---  
## Overall Workflow  
  
URDF kitchen consists of the following **4 steps**:  
  
0. **Mesh Export** : Export parts from your CAD or modeling tool  
1. **Mesh Sourcer** : Adjust mesh origin, axes, and colliders  
2. **Parts Editor** : Define connection points, inertia, and other per-part properties  
3. **Assembler** : Connect nodes and export as URDF/MJCF  
  
---  
## Sample  
  
**Pre-configured mesh and XML files are stored in the sample/Roid1_assets directory.**  
From STEP 1 onward, these sample files will be used as examples.  
  
---  
## STEP 0: Mesh Export  
  
First, export your robot as **mesh files** (STL/OBJ/DAE), one per unit.  
This step is typically done in your own CAD or 3D modeling software.  
  
Here, a "unit" refers to a group consisting of a rotation axis (joint) and a body (link) that move together.  
For example, a forearm unit would pair the elbow as the rotation axis (joint) with the forearm body extending to the wrist as the link.  
In this guide, each unit is represented by one mesh file.  
  
### Use a Consistent Coordinate System  
Different modeling tools use different coordinate systems. Export your mesh in a standard right-handed coordinate system used by simulators:  
  
| Axis | Direction |  
| --- | --- |  
| X   | Forward |  
| Y   | Left |  
| Z   | Up |  
  
### Check 1: Unit Origin = Rotation Origin  
When exporting a forearm as a mesh, make sure the following conditions are met:  
- The model's origin is the center of rotation  
- That origin serves as the connection point from the parent unit  
  
### Check 2: Confirm Child Unit Connection Points  
Keep track of the points (coordinates) where child units will be connected.  
You can mark these on the model itself, establish reference landmarks, or simply note down the coordinates (in meters).  
  
### Check 3: Mesh Naming Convention  
When exporting models, follow these naming rules:  
- Units belonging to the **left side** of the body must have **"l_"** at the beginning of the file name  
- Units on the **center line** of the body must have **"c_"** at the beginning of the file name  
For example, the left shoulder would be "l_shoulder_pitch", and the head would be "c_head".  
Right-side units use "r_", but if the robot is left-right symmetric, you do not need to create them manually — they can be auto-generated.  
  
You can also define decorative mesh files that carry no physical properties. In that case, append **"_dec1"** to the end of the file name. If there are multiple decorations for the same part, increment the number (e.g., _dec2, _dec3).  
  
### Check 4: File Format and Units  
URDF typically references meshes in STL, OBJ, and DAE (Collada).  
Set the export unit to **meters**.  
  
### Check 5: Output Directory  
Save all mesh files together in a single directory.  
Batch operations will process all files in the top level of that directory.  
  
Once all **left-side and center units** have been exported, this step is complete.  
**If the robot is symmetric, right-side units are not needed** — they will be auto-generated via mirror copy later.  
  
To get a sense of the overall workflow, you can move on once you have at least two or more consecutive parts ready.  
Now it is time to start using URDF kitchen.  
  
  
  
---  
## STEP 1: Mesh Sourcer  
  
In this step, you can set up Colliders (collision geometry) based on the exported unit mesh files.  
You can also adjust the mesh origin, configure axes, and reduce mesh complexity.  
If your mesh files are already set up correctly and you plan to use the mesh itself as the collider, you can skip this step and proceed directly to STEP 2.  
  
```  
python urdf_kitchen_Launcher.py  
```  
  
Launch the application and click the Mesh Sourcer button.  
  
### 1-1. Mesh Sourcer Basic Controls  
  
Click the **"Load Mesh"** button to select a mesh file to load.  
  
```  
Example: Select c_chest.stl from the sample/Roid1_assets directory  
```  
  
The 3D view on the left side supports the following controls.  
These controls are shared across Parts Editor and Assembler as well.  
  
| Action | Key |  
| --- | --- |  
| Rotate left/right | A / D |  
| Rotate up/down | W / S |  
| Roll clockwise/counterclockwise | Q / E |  
| Reset view | R |  
| Wireframe toggle | T (ON/OFF) |  
| Move marker | Arrow keys (relative to the screen) |  
| Fine adjustment | Shift / Ctrl + Arrow keys |  
| Direct numeric input | Type values directly |  
| Drag | Rotate the camera |  
| Wheel + Drag | Pan the camera |  
| Wheel | Zoom in/out |  
  
### 1-2. Origin Coordinates (Origin & Axes)  
  
Use this when you need to adjust the mesh origin coordinates or swap coordinate axes.  
  
With the Target Marker Position checkbox enabled, use the arrow keys to move the purple marker that sets the new origin. The marker moves in 10 mm increments relative to the screen, 1 mm with Shift held, and 0.1 mm with Ctrl held. You can also type coordinates directly into the input fields.  
Rotate the 3D view with A/S/D/W and move the marker to the desired origin position.  
  
- **Reset Marker** : Resets the marker position to 0, 0, 0.  
- **Set Front as X** : Reassigns the coordinate axes so that the current front-facing direction in the 3D view becomes the X axis.  
- **Save** : Saves the mesh with the current axis orientation and marker position as the new origin.  
- **Clean Mesh** : When checked, cleans up and simplifies the mesh data.  
  
### 1-3. Collider Design  
  
You can define the collision region (Collider) using simple shapes called Primitives.  
Using Primitives instead of the full mesh for collision detection reduces the computational load on the simulator.  
  
Pressing "T" to switch to wireframe mode is helpful when configuring colliders.  
  
- **Show Collider** : Displays the collider region.  
- **Type** : Choose the collider shape: Box, Sphere, Cylinder, or Capsule.  
- **Position** : Move the center of the collider. Can be adjusted with arrow keys.  
- **Size** : Change the collider dimensions. Can be adjusted with arrow keys.  
- **Rotation** : Adjust the collider orientation. Left/right arrows rotate clockwise/counterclockwise relative to the screen.  
  
While configuring colliders, press TAB to cycle between Position, Size, and Rotation modes.  
  
- **Rough Fit** : Automatically assigns an approximate collider to the mesh.  
- **Reset Collider** : Resets the collider settings.  
- **Export Collider** : Exports the collider as an XML file.  
  
Once colliders are properly configured, use Export Collider to save the settings. A file named mesh_name_collider.xml will be saved in the same directory as the loaded mesh.  
  
```  
Example: If the mesh is c_chest.stl, a file named c_chest_collider.xml is exported.  
```  
  
### 1-4. Batch Mesh Converter  
  
Converts mesh file formats in batch for an entire directory.  
  
- **Input** : Select the source file format.  
- **Output** : Select the target file format.  
- **Clean Mesh** : When checked, cleans up and simplifies the mesh data.  
- **Select Directory and Convert** : Select a directory and execute the conversion.  
  
  
Once the necessary colliders have been configured for each unit, this step is complete.  
You do not need to create colliders for every single unit — even setting them for just the hands and feet is enough for the simulator to function at a basic level.  
(However, if there are no colliders at all, the robot will fall through the floor, so keep that in mind.)  
  
---  
## STEP 2: Parts Editor  
  
In this step, you define the coordinates for connecting child units to each unit's mesh.  
You can also configure weight, inertia tensor, rotation axis, color, and other properties.  
The configured parameters are saved as an XML file named mesh_name.xml.  
  
### 2-1. Parts Editor Basic Controls  
  
The 3D view on the right side operates similarly to Mesh Sourcer.  
The settings panel on the left is arranged in three sections following the workflow order:  
the top section for loading unit mesh files, the middle for parameter settings, and the bottom for export and conversion.  
  
### 2-2. Loading Unit Mesh Files  
  
- **Import Mesh** : Loads a mesh file.  
- **Load XML** : Loads an XML file created by Parts Editor.  
- **Reload** : Reloads the XML contents.  
- **Load Mesh with XML** : Loads both a mesh and its corresponding XML file together.  
  
When a mesh is loaded, the model appears in the 3D view along with a **red sphere indicating the center of mass**. Pressing T to switch to wireframe mode makes the center of mass easier to see.  
  
```  
Example: Click Import Mesh and load c_chest.stl.  
         Press T to switch the 3D view to wireframe mode, revealing the red center-of-mass sphere.  
```  
### 2-3. Mass and Inertia Settings  
  
- **MeshSourcer** : Opens the current mesh in Mesh Sourcer to go back and edit it.  
- **Volume** : The volume of the unit.  
- **Density** : The density of the unit.  
- **Mass** : The mass of the unit in kg.  
- **Center of Mass** : The center-of-mass coordinates of the unit.  
- **Inertia Tensor** : The inertia tensor values.  
- **Zero off-diag** : Performs a simplified diagonalization of the computed inertia tensor.  
- **Calculate** : Computes the inertia tensor and related values.  
  
The "Calculate" button computes results, treating checked items as fixed values and automatically calculating unchecked items along with the inertia tensor.  
A typical workflow is: enter a measured value for Mass and check it, optionally enter approximate center-of-mass coordinates for Center of Mass and check it, check Zero off-diag, then click Calculate. Volume and density are automatically computed as reference values, and the inertia tensor is calculated based on the fixed values.  
If Center of Mass is left unchecked when you click Calculate, the center of mass is computed from the mesh data.  
  
When Center of Mass is checked, a red marker appears in the 3D view. The marker can be moved with arrow keys, but be sure to uncheck it after Calculate is done. (If left checked, the marker will move in sync with Point markers described later.)  
  
```  
Example: Check Mass and enter "0.4215".  
         Check Center of Mass and enter 0 for Y.  
         Check Zero off-diag.  
         Click Calculate.  
         Result: the inertia tensor is computed with the specified mass and center of mass as fixed values, and diagonalized.  
  
         Mass(kg): 0.4215  
         Center of Mass: X:-0.016071, Y:0, Z:0.032638  
         <inertia ixx="0.00069558" ixy="0.00000000" ixz="0.00000000"  
          iyy="0.00084151" iyz="0.00000000" izz="0.00088363"/>  
```  
  
### 2-4. Color Settings  
  
You can set the color of the unit. Enter values directly in the input fields, or click the Pick button to open a color palette.  
  
### 2-5. Rotation Axis Settings  
  
Configure the rotation direction of the unit around its mesh origin.  
- Axis : Choose the rotation direction from X:roll, Y:pitch, or Z:yaw. Select Fixed if the unit does not rotate.  
- Rotate Test : Preview the rotation in the 3D view.  
  
```  
Example: Check Z:yaw for Axis.  
         Press and hold Rotate Test — the model rotates in the 3D view around the Z axis through the origin.  
```  
  
### 2-6. Child Unit Connection Points  
  
Define the coordinates where child units will be connected. Multiple connection points can be set. For example, a waist part might have connection points for the right hip, left hip, and upper body.  
  
- **Angle(deg)** : Specifies the angular offset of the rotation axis for the checked connection point.  
- **Point 1~8** : Connection point coordinates. When the left-side checkbox is checked, a marker appears and can be moved with arrow keys. You can also type values directly into the input fields.  
- **Reset Point** : Resets the values of checked Points.  
  
You can check multiple Points to move their markers simultaneously, or copy a Y value and flip its sign for symmetric left/right configurations.  
  
```  
Example: Point1: X:0.016030, Y:0.000000,  Z:0.062400  
         Point2: X:0.016030, Y:0.048855,  Z:0.045405  
         Point3: X:0.016030, Y:-0.048855, Z:0.045405  
         Angle(deg) is left unset.  
```  
  
### 2-7. XML File Export  
  
Export the configured parameters as an XML file.  
- **Export XML** : Exports the settings as an XML file.  
- **Export Mirror Mesh with XML** : Exports a single mirrored file.  
- **Batch Mirror "l_" to "r_" Meshes and XMLs** : Batch-creates mirrored mesh and XML files for all units in the specified directory.  
  
In this step as well, the work is complete once all **left-side and center units** have been exported.  
Finally, use **Batch Mirror "l_" to "r_" Meshes and XMLs** to generate all right-side units at once.  
  
```  
Example: Click Export XML. The XML file is saved in the same directory as the mesh.  
```  
  
```  
Example: Click Batch Mirror "l_" to "r_" Meshes and XMLs and  
         select the Roid1_assets directory.  
         Right-side files are mirror-generated from the left-side mesh and XML files in the sample data.  
```  
  
This completes the preparation of all units. Now comes the fun part — assembly!  
  
---  
  
## STEP 3: Assembler  
  
Load the unit mesh files and their parameter XML files into the Assembler's node panel. By connecting these nodes together, you assemble the robot and can export it as a URDF or MJCF file. Any parameters not yet configured can also be set here.  
  
### 3-1. Starting the Assembler  
  
Click the Assembler button in the Launcher to open the window.  
The left side is the navigation panel, the center is the node view, and the right side is the 3D view.  
Both the node view and 3D view can be zoomed with the scroll wheel.  
The borders between panels can be dragged to resize them.  
  
Here are a few buttons to start with:  
  
- **Add Node** : Creates a new node.  
- **Delete Node** : Deletes the selected node.  
- **Recalc Positions** : Recalculates unit display positions in the 3D view.  
  
Other buttons will be explained as they come up in the workflow.  
  
### 3-2. Loading XML Files  
  
Load the units you have configured so far.  
  
- **Import XMLs** : Specify a directory to load all XML files along with their associated mesh files.  
- **Import MODEL** : Import a URDF or MJCF file. Details are described later.  
  
When you use **Import XMLs**, all configuration files are loaded at once. Node panels appear in the main window and unit mesh files are displayed in the 3D view. Since nodes are not yet connected at this point, all unit mesh files appear clustered at the center.  
  
```  
Example: Click Import XMLs and select the Roid1_assets directory.  
         Nodes corresponding to each unit appear on the screen.  
         Unconnected nodes are displayed in gray, so all nodes except base_link will be gray.  
```  
  
### 3-3. Assembling by Connecting Nodes  
  
First, connect the root node (e.g., the waist) to the "base_link" node in the upper left. Once a node is connected to base_link, it changes from gray to black.  
Next, connect that node's out port (orange dot) to the corresponding child node's in port (orange dot). The assembly result is then displayed in the 3D view.  
Coordinates are only applied to nodes connected to base_link; units belonging to unconnected nodes are displayed at the origin.  
  
```  
Example: Connect base_link's out port to c_waist's in port.  
         Connect c_waist's out_1 port to c_chest's in port.  
         Connect c_waist's out_2 port to l_hipjoint_upper's in port.  
         Connect c_chest's out_1 port to c_head's in port.  
         Connect c_chest's out_2 port to l_shoulder's in port.  
```  
  
### 3-4. Node Inspector: Working with Out Ports  
  
Double-click a node to open the Node Inspector.  
Here you can configure various parameters.  
To start, let's add out ports and connect decorative units.  
Use the Add outport and Remove outport buttons at the bottom of the Node Inspector.  
  
```  
Example: Double-click the c_chest node to open the Node Inspector.  
         Click Add outport twice to create outport_4 and outport_5.  
         Close the Node Inspector.  
         Connect c_chest's outport_4 to c_chest_dec1.  
         Connect c_chest's outport_5 to c_chest_dec2.  
         Verify in the 3D view that the chest section has been updated.  
```  
  
### 3-5. Node Inspector: Color Settings  
  
You can specify the unit's color in the Node Inspector.  
Click the "Pick" button on the color: row to open a color palette where you can freely choose colors.  
To save a color for reuse, click a box in the Custom colors area of the palette, pick a color, and press the Add to Custom colors button. This is convenient when applying the same color to multiple parts.  
  
```  
Example: Double-click the c_chest_dec1 node to open the Node Inspector.  
         Click the "Pick" button on the color: row in the middle of the Node Inspector.  
         Select any box in the Custom colors area of the color palette window.  
         Pick a color and click Add to Custom colors to save it.  
         Click OK to close the palette — the color is applied to the unit in the 3D view.  
```  
  
### 3-6. Save / Load Operations  
  
You can save and restore your work progress.  
- Save Project : Saves the current work. The file name includes a timestamp.  
- Load Project : Loads a previously saved project.  
  
```  
Example: Click Save Project and save the project file to a convenient directory.  
         Close the Assembler completely — either by clicking the window's close button  
         or pressing Ctrl+C in the terminal/command prompt.  
         Relaunch the Assembler, click Load Project, and select the saved project file.  
         Verify that the work is fully restored.  
```  
  
### 3-7. Assembling the Left Side and Center  
  
This part requires some effort — assemble all left-side and center parts here.  
It helps to select the right-side parts and move them off to the side to keep the workspace clear.  
Arranging connected nodes in an organized pattern also makes things easier to follow.  
By default, nodes snap to a 50-pixel grid on the screen.  
You can also drag the borders of the navigation panel and 3D view to collapse them, giving the node view more workspace.  
  
  
```  
Example: The node connections are as follows:  
  
base_link - c_waist_out1 - c_chest  
                  ├_out2 - l_hipjoint_upper  
                  └_out3  
  
c_chest_out1 - c_head  
      ├_out2 - l_shoulder  
      └_out3  
  
l_shoulder - l_arm_upper - l_elbow - l_arm_lower  
  
l_hipjoint_upper - l_hipjoint_lower - l_leg_upper - l_leg_lower - l_ankle - l_foot  
  
Decorative connections:  
  
c_chest_out4 - c_chest_dec1  
c_chest_out5 - c_chest_dec2  
l_shoulder_out2 - l_shoulder_dec1  
l_arm_upper_out2 - l_arm_upper_dec1  
l_hipjoint_upper_out2 - l_hipjoint_upper_dec1  
l_leg_upper_out2 - l_leg_upper_dec1  
l_leg_lower_out2 - l_leg_lower_dec1  
  
Save the project once all connections are complete.  
  
```  
  
### 3-8. Node Inspector: Loading and Saving Node Information  
  
You can load and save information for individual Node Inspectors.  
  
Buttons at the top of the window:  
- **Import Mesh** : Loads a mesh file.  
- **Load XML** : Loads a settings XML file.  
- **Load XML with Mesh** : Specify an XML file to load it along with its associated mesh.  
- **Reload** : Refreshes the information based on the saved XML file.  
  
Button at the bottom of the window:  
- **Save XML** : Saves the individual Node Inspector's information.  
  
### 3-9. Node Inspector: Node Attribute Settings  
  
Use the checkboxes in the window to configure node attributes.  
  
- **Massless Decoration** : When checked, the node is treated as a visual-only element with no physical properties on export.  
- **Hide Mesh** : Hides the unit in the 3D view.  
  
```  
Example: Set Hide Mesh for the following nodes:  
         l_foot_small, l_foot_large  
```  
### 3-10. Node Inspector: Mass and Inertia Settings  
  
Similar to Parts Editor, you can perform simplified inertia configuration.  
  
Button descriptions:  
- **Parts Editor** : Opens Parts Editor so you can go back and work there.  
- **Show CoM** : Displays the center-of-mass coordinates in the 3D view.  
- **Recalc CoM** : Automatically recalculates the center of mass from mesh data.  
- **Recalc Inertia** : Recalculates the inertia tensor.  
- **Zero off-diag** : Performs a simplified diagonalization of the inertia tensor result.  
  
### 3-11. Node Inspector: Rotation Axis and Range of Motion  
  
Configure the rotation angle and range of motion for the unit corresponding to each node.  
This refers to the unit's own rotation, with the mesh origin as the axis.  
  
- **Rotation Axis** : Change the rotation axis.  
- **Angle offset(deg)** : Sets the rotation axis offset angle at the mesh origin. This value is shared with the Ang value of the parent node's out port.  
- **Min Angle (deg), Max Angle(deg)** : Set the minimum and maximum rotation angles.  
- **Show Min** : Displays the minimum rotation angle in the 3D view.  
- **Show Max** : Displays the maximum rotation angle in the 3D view.  
- **Show Zero** : Displays the rotation origin (zero position) in the 3D view.  
- **Rotation Test** : While the button is held, the rotation range is animated in the 3D view.  
  
### 3-12. Node Inspector: Actuator Parameter Settings  
  
You can configure actuator characteristics for MJCF and similar formats.  
Regardless of whether the actuator belongs to the parent or the current node, these settings apply to the rotational axis (joint) connecting the parent and current node.  
Note: As of Beta 2, this feature has not been fully validated — please use it as a reference only.  
  
| Parameter | MJCF `<joint>` | MJCF `<actuator>` | MuJoCo Physical Meaning |  
| --- | --- | --- | --- |  
| Effort | — | forcerange | Maximum actuator torque |  
| Damping (kv) | (not output to MJCF) | kv | Viscous damping (passive) + PD control D gain (active) |  
| Stiffness (kp) | (not output to MJCF) | kp | PD control P gain (active only) |  
| Velocity | (not output to MJCF) | — | Not used in MJCF export |  
| Margin | margin | — | Soft zone width for limit constraints |  
| Armature | armature | — | Diagonal addition to the inertia matrix (numerical stability + rotor inertia) |  
| Frictionloss | frictionloss | — | Dry friction (Coulomb friction) torque |  
  
### 3-13. Node Inspector: Collider Settings  
  
You can assign either a mesh or a simple shape called a Primitive as the Collider.  
Primitive colliders are configured in Mesh Sourcer as described earlier.  
  
- **Mesh Sourcer** : Opens Mesh Sourcer for further work.  
- **Colliders checkbox** : Check to enable the collider; uncheck to disable it.  
- **Attach** : Attach a mesh or an "x_collider.xml" file.  
- **+/-** : Add or remove colliders when multiple colliders are needed for a single node.  
  
```  
Example: Uncheck the Colliders checkbox for the following nodes:  
         l_foot_small, l_foot_large  
```  
  
### 3-14. Creating the Right Side via Mirror  
  
You can automatically create the right side by mirroring the left-side node data.  
Connections and node panel positions are also automatically arranged based on the left side.  
  
- **Build r_ from l_** : Click this button to perform the automatic right-side assembly.  
  
This completes the model.  
Any unconnected node panels are displayed in gray — connect them to the appropriate parent node.  
If they are not needed, use Hide Mesh or delete the node to handle them.  
  
### 3-15. Exporting URDF or MJCF  
  
Export the robot model.  
  
- **Export URDF** : Exports a URDF file. Creates a directory containing the URDF file and mesh files.  
- **Export for Unity** : Exports in a format compatible with Unity's URDF-importer.  
- **Export MJCF** : Exports as MuJoCo files, including a scene file.  
  
### 3-16. Verification  
  
After exporting URDF, you can click the **Open urdf-loaders** button to open California Institute of Technology's urdf-loaders in your browser. Drag and drop the entire exported description directory into the browser to view the model.  
  
After exporting MJCF, drag and drop the generated scene.xml file into the MuJoCo viewer to open the model.  
  
If anything looks wrong with the assembly, go back to the Assembler or earlier steps to fix the model.  
  
Congratulations — you are done!  
  
## Import MODEL  
  
As a bonus feature, you can open existing URDF or MJCF files.  
Click the **Import MODEL** button in the Assembler to open a dialog box.  
  
- **Import URDF/SDF** : Opens a URDF-based model. Select any .urdf file. **Xacro** files are also supported. When opening an SDF file, you will also need to re-select the base URDF.  
- **Import MJCF** : Opens a MuJoCo file. Except for closed-loop linkages, models are reproduced in the Assembler with reasonable accuracy.  
  
## Final Notes  
  
Thank you for trying URDF kitchen. We hope this application helps you create models for your custom robots.  
As this is a beta version, there are likely many bugs. We welcome your feedback and reports.  
Since the application is written in Python, it is easy to modify or extend with AI-assisted coding. Feel free to fork the project and build your own version of URDF kitchen.  
