extends Node3D
## B2 robot visualization with primitive shapes.
##
## Body is a box, legs are chains of Node3D pivots with capsule meshes.
## Visible joint spheres at hip, knee, and foot make articulation clear.
## Joints arrive in MuJoCo body order: FL,FR,RL,RR x hip,thigh,calf.
##
## MuJoCo joint axes (from b2.xml):
##   hip:   axis 1,0,0 (MuJoCo X = Godot X) — abduction
##   thigh: axis 0,1,0 (MuJoCo Y = Godot -Z) — flexion
##   calf:  axis 0,1,0 (MuJoCo Y = Godot -Z) — flexion
##
## Coordinate mapping: MuJoCo (x,y,z) -> Godot (x,z,-y)
## The Y-negation preserves right-handedness (without it the view is mirrored).

# B2 dimensions (from MuJoCo XML)
const BODY_SIZE := Vector3(0.50, 0.15, 0.28)
const THIGH_LEN := 0.35
const CALF_LEN := 0.35
const THIGH_R := 0.025
const CALF_R := 0.02
const FOOT_R := 0.035
const JOINT_R := 0.035  # Visible joint sphere radius

# Hip offsets from body center: MuJoCo (x,y) -> Godot (x,-z)
const HIP_OFFSETS := {
	"FL": Vector3(0.33, 0.0, -0.07),
	"FR": Vector3(0.33, 0.0, 0.07),
	"RL": Vector3(-0.33, 0.0, -0.07),
	"RR": Vector3(-0.33, 0.0, 0.07),
}
# Hip-to-thigh lateral offset (MuJoCo Y -> Godot -Z)
const HIP_LATERAL := {
	"FL": -0.12, "FR": 0.12,
	"RL": -0.12, "RR": 0.12,
}

# Colors
const BODY_COLOR := Color(0.3, 0.35, 0.45)
const THIGH_COLOR := Color(0.45, 0.50, 0.60)
const CALF_COLOR := Color(0.55, 0.58, 0.65)
const JOINT_COLOR := Color(0.85, 0.55, 0.15)  # Orange-gold for joints
const FOOT_COLOR := Color(0.7, 0.7, 0.7)

var _body: MeshInstance3D
var _legs: Dictionary = {}
const LEG_ORDER := ["FL", "FR", "RL", "RR"]


func _ready() -> void:
	_create_body()
	for leg_name in LEG_ORDER:
		_create_leg(leg_name)


func _create_body() -> void:
	_body = MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = BODY_SIZE
	_body.mesh = box
	var mat := StandardMaterial3D.new()
	mat.albedo_color = BODY_COLOR
	_body.material_override = mat
	add_child(_body)


func _create_joint_sphere(color: Color) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = JOINT_R
	sphere.height = JOINT_R * 2
	mi.mesh = sphere
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.emission_enabled = true
	mat.emission = color
	mat.emission_energy_multiplier = 0.3
	mi.material_override = mat
	return mi


func _create_capsule(radius: float, length: float, color: Color) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var cap := CapsuleMesh.new()
	cap.radius = radius
	cap.height = length + radius * 2
	mi.mesh = cap
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mi.material_override = mat
	return mi


func _create_leg(leg_name: String) -> void:
	var hip_offset: Vector3 = HIP_OFFSETS[leg_name]
	var lat: float = HIP_LATERAL[leg_name]

	# Hip pivot — rotates around Godot X (MuJoCo hip axis 1,0,0)
	var hip := Node3D.new()
	hip.position = hip_offset
	hip.name = leg_name + "_Hip"
	add_child(hip)

	# Hip joint sphere (visible at hip pivot)
	var hip_sphere := _create_joint_sphere(JOINT_COLOR)
	hip.add_child(hip_sphere)

	# Thigh pivot — rotates around Godot Z (MuJoCo thigh axis 0,1,0)
	var thigh_pivot := Node3D.new()
	thigh_pivot.position = Vector3(0, 0, lat)
	thigh_pivot.name = leg_name + "_ThighPivot"
	hip.add_child(thigh_pivot)

	var thigh_mesh := _create_capsule(THIGH_R, THIGH_LEN, THIGH_COLOR)
	thigh_mesh.position = Vector3(0, -THIGH_LEN / 2.0, 0)
	thigh_pivot.add_child(thigh_mesh)

	# Calf pivot — rotates around Godot Z (MuJoCo calf axis 0,1,0)
	var calf_pivot := Node3D.new()
	calf_pivot.position = Vector3(0, -THIGH_LEN, 0)
	calf_pivot.name = leg_name + "_CalfPivot"
	thigh_pivot.add_child(calf_pivot)

	# Knee joint sphere (visible at thigh-calf junction)
	var knee_sphere := _create_joint_sphere(JOINT_COLOR)
	calf_pivot.add_child(knee_sphere)

	var calf_mesh := _create_capsule(CALF_R, CALF_LEN, CALF_COLOR)
	calf_mesh.position = Vector3(0, -CALF_LEN / 2.0, 0)
	calf_pivot.add_child(calf_mesh)

	# Foot sphere
	var foot := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = FOOT_R
	sphere.height = FOOT_R * 2
	foot.mesh = sphere
	var foot_mat := StandardMaterial3D.new()
	foot_mat.albedo_color = FOOT_COLOR
	foot.material_override = foot_mat
	foot.position = Vector3(0, -CALF_LEN, 0)
	calf_pivot.add_child(foot)

	_legs[leg_name] = {
		"hip": hip,
		"thigh": thigh_pivot,
		"calf": calf_pivot,
	}


func update_pose(
	slam_x: float, slam_y: float, slam_yaw: float,
	z: float, roll: float, pitch: float,
	joints: Array[float]
) -> void:
	# MuJoCo (x,y,z) -> Godot (x,z,-y). Negate Y to preserve handedness.
	# Yaw stays positive = CCW from above. Pitch negated (axis flipped).
	position = Vector3(slam_x, z, -slam_y)
	rotation = Vector3(roll, slam_yaw, -pitch)

	# Apply joint angles per leg
	for i in 4:
		var leg_name: String = LEG_ORDER[i]
		var hip_q: float = joints[i * 3]      # hip abduction
		var thigh_q: float = joints[i * 3 + 1] # thigh flexion
		var calf_q: float = joints[i * 3 + 2]  # calf flexion
		var leg: Dictionary = _legs[leg_name]

		# Hip: MuJoCo axis (1,0,0) = Godot X rotation
		leg["hip"].rotation.x = hip_q
		# Thigh: MuJoCo axis (0,1,0) -> Godot Z rotation
		leg["thigh"].rotation.z = thigh_q
		# Calf: MuJoCo axis (0,1,0) -> Godot Z rotation
		leg["calf"].rotation.z = calf_q
