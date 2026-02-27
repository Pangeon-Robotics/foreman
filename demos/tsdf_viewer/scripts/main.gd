extends Node3D
## Main scene controller — wires TCP signals to renderers.

@onready var tcp: Node = $TCPClient
@onready var robot: Node3D = $Robot
@onready var tsdf_voxels: Node3D = $TSDFVoxels
@onready var astar_path: Node3D = $AStarPath
@onready var obs_overlay: MeshInstance3D = $ObservationOverlay
@onready var obstacle_renderer: Node3D = $ObstacleRenderer
@onready var target_mesh: MeshInstance3D = $Target
@onready var status_label: Label = $HUD/StatusLabel
@onready var conn_label: Label = $HUD/ConnLabel
@onready var camera: Camera3D = $Camera3D

# Latest parsed state for HUD
var _dist := 0.0
var _heading_err := 0.0
var _n_feasible := 0
var _dwa_fwd := 0.0
var _dwa_turn := 0.0
var _game_state := 0
var _in_tip := false


func _ready() -> void:
	# Create target mesh programmatically
	var sphere := SphereMesh.new()
	sphere.radius = 0.3
	sphere.height = 0.6
	sphere.radial_segments = 24
	target_mesh.mesh = sphere
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1.0, 0.85, 0.2, 1.0)
	mat.emission_enabled = true
	mat.emission = Color(1.0, 0.85, 0.2)
	mat.emission_energy_multiplier = 0.5
	target_mesh.material_override = mat

	# Connect TCP signals
	tcp.connected.connect(_on_connected)
	tcp.disconnected.connect(_on_disconnected)
	tcp.robot_state_received.connect(_on_robot_state)
	tcp.tsdf_received.connect(_on_tsdf)
	tcp.path_received.connect(_on_path)
	tcp.observation_map_received.connect(_on_obs_map)
	tcp.obstacles_received.connect(_on_obstacles)


func _on_connected() -> void:
	conn_label.text = "Connected"
	conn_label.add_theme_color_override("font_color", Color.GREEN)


func _on_disconnected() -> void:
	conn_label.text = "Disconnected — reconnecting..."
	conn_label.add_theme_color_override("font_color", Color.RED)


func _on_robot_state(data: PackedByteArray) -> void:
	if data.size() < 100:
		return
	var ofs := 0
	var slam_x := data.decode_float(ofs); ofs += 4
	var slam_y := data.decode_float(ofs); ofs += 4
	var slam_yaw := data.decode_float(ofs); ofs += 4
	var z := data.decode_float(ofs); ofs += 4
	var roll := data.decode_float(ofs); ofs += 4
	var pitch := data.decode_float(ofs); ofs += 4

	# 12 joints (MuJoCo body order: FL,FR,RL,RR x hip,thigh,calf)
	var joints: Array[float] = []
	for i in 12:
		joints.append(data.decode_float(ofs)); ofs += 4

	var target_x := data.decode_float(ofs); ofs += 4
	var target_y := data.decode_float(ofs); ofs += 4
	_game_state = data[ofs]; ofs += 1
	_dwa_fwd = data.decode_float(ofs); ofs += 4
	_dwa_turn = data.decode_float(ofs); ofs += 4
	_n_feasible = data.decode_u16(ofs); ofs += 2
	_heading_err = data.decode_float(ofs); ofs += 4
	_dist = data.decode_float(ofs); ofs += 4
	_in_tip = data[ofs] != 0

	# MuJoCo Z-up -> Godot Y-up: (x,y,z) -> (x,z,-y)
	robot.update_pose(slam_x, slam_y, slam_yaw, z, roll, pitch, joints)

	# Update target position
	target_mesh.visible = (_game_state == 3)  # WALK_TO_TARGET
	target_mesh.position = Vector3(target_x, 0.3, -target_y)

	# Camera follow
	camera.set_follow_target(Vector3(slam_x, 0, -slam_y))


func _on_tsdf(data: PackedByteArray) -> void:
	tsdf_voxels.update_tsdf(data)


func _on_path(data: PackedByteArray) -> void:
	astar_path.update_path(data)


func _on_obs_map(data: PackedByteArray) -> void:
	obs_overlay.update_observation_map(data)


func _on_obstacles(data: PackedByteArray) -> void:
	obstacle_renderer.update_obstacles(data)


func _process(_delta: float) -> void:
	# Update HUD
	var state_name := "?"
	match _game_state:
		1: state_name = "STARTUP"
		2: state_name = "SPAWN"
		3: state_name = "WALK"
		4: state_name = "DONE"
	var tip_str := " [TIP]" if _in_tip else ""
	status_label.text = (
		"State: %s%s\n" % [state_name, tip_str] +
		"Dist: %.2fm  Heading: %.1f deg\n" % [_dist, rad_to_deg(_heading_err)] +
		"DWA: fwd=%.2f  turn=%.2f  feas=%d\n" % [_dwa_fwd, _dwa_turn, _n_feasible]
	)
