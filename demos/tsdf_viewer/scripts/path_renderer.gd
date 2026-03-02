extends Node3D
## Renders A* path waypoints as colored spheres using MultiMesh.
##
## Robot-view path: GREEN spheres (existing)
## God-view path:   BLUE spheres (new)

var _multi_mesh_inst: MultiMeshInstance3D
var _multi_mesh: MultiMesh

var _god_multi_mesh_inst: MultiMeshInstance3D
var _god_multi_mesh: MultiMesh


func _ready() -> void:
	# --- Green (robot-view) path ---
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.2, 0.9, 0.3)
	mat.emission_enabled = true
	mat.emission = Color(0.1, 0.6, 0.15)
	mat.emission_energy_multiplier = 0.3

	var sphere := SphereMesh.new()
	sphere.radius = 0.08
	sphere.height = 0.16
	sphere.radial_segments = 8

	_multi_mesh = MultiMesh.new()
	_multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	_multi_mesh.mesh = sphere
	_multi_mesh.instance_count = 0

	_multi_mesh_inst = MultiMeshInstance3D.new()
	_multi_mesh_inst.multimesh = _multi_mesh
	_multi_mesh_inst.material_override = mat
	add_child(_multi_mesh_inst)

	# --- Blue (god-view) path ---
	var god_mat := StandardMaterial3D.new()
	god_mat.albedo_color = Color(0.2, 0.3, 0.9)
	god_mat.emission_enabled = true
	god_mat.emission = Color(0.1, 0.15, 0.6)
	god_mat.emission_energy_multiplier = 0.3

	var god_sphere := SphereMesh.new()
	god_sphere.radius = 0.08
	god_sphere.height = 0.16
	god_sphere.radial_segments = 8

	_god_multi_mesh = MultiMesh.new()
	_god_multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	_god_multi_mesh.mesh = god_sphere
	_god_multi_mesh.instance_count = 0

	_god_multi_mesh_inst = MultiMeshInstance3D.new()
	_god_multi_mesh_inst.multimesh = _god_multi_mesh
	_god_multi_mesh_inst.material_override = god_mat
	add_child(_god_multi_mesh_inst)


func update_path(data: PackedByteArray) -> void:
	_apply_path_data(data, _multi_mesh)


func update_god_view_path(data: PackedByteArray) -> void:
	_apply_path_data(data, _god_multi_mesh)


func _apply_path_data(data: PackedByteArray, mm: MultiMesh) -> void:
	# u16 n_points, then f32x2 per point
	if data.size() < 2:
		return
	var n: int = data.decode_u16(0)
	var expected: int = 2 + n * 8
	if data.size() < expected:
		return

	mm.instance_count = n
	for i in n:
		var base: int = 2 + i * 8
		var wx := data.decode_float(base)
		var wy := data.decode_float(base + 4)
		# MuJoCo (x,y) -> Godot (x, 0.15, -y)
		var t := Transform3D.IDENTITY
		t.origin = Vector3(wx, 0.15, -wy)
		mm.set_instance_transform(i, t)
