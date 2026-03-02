extends Node3D
## Renders TSDF occupied voxels as 3D cubes at their actual positions.
##
## Each occupied voxel is rendered as a cube scaled to the actual voxel_size
## (1cm voxels). When the LiDAR provides
## multi-elevation data, voxels appear at multiple heights showing the
## true 3D structure of obstacles.

var _multi_mesh_inst: MultiMeshInstance3D
var _multi_mesh: MultiMesh


func _ready() -> void:
	var mat := StandardMaterial3D.new()
	mat.vertex_color_use_as_albedo = true
	mat.emission_enabled = true
	mat.emission_energy_multiplier = 0.5

	# Unit cube — scaled per-instance by actual voxel_size from data
	var box := BoxMesh.new()
	box.size = Vector3(1.0, 1.0, 1.0)

	_multi_mesh = MultiMesh.new()
	_multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	_multi_mesh.use_colors = true
	_multi_mesh.mesh = box
	_multi_mesh.instance_count = 0

	_multi_mesh_inst = MultiMeshInstance3D.new()
	_multi_mesh_inst.multimesh = _multi_mesh
	_multi_mesh_inst.material_override = mat
	add_child(_multi_mesh_inst)


func update_tsdf(data: PackedByteArray) -> void:
	if data.size() < 24:
		return

	var ofs := 0
	var origin_x := data.decode_float(ofs); ofs += 4
	var origin_y := data.decode_float(ofs); ofs += 4
	var z_min := data.decode_float(ofs); ofs += 4
	var voxel_size := data.decode_float(ofs); ofs += 4
	@warning_ignore("unused_variable")
	var nx := data.decode_u16(ofs); ofs += 2
	@warning_ignore("unused_variable")
	var ny := data.decode_u16(ofs); ofs += 2
	var n_voxels := data.decode_u32(ofs); ofs += 4

	var expected_size: int = ofs + n_voxels * 6
	if data.size() < expected_size:
		return

	_multi_mesh.instance_count = n_voxels
	var half_vs := voxel_size * 0.5

	for i in n_voxels:
		var base: int = ofs + i * 6
		var ix := data.decode_u16(base)
		var iy := data.decode_u16(base + 2)
		var iz := data[base + 4]
		var lo_raw := data[base + 5]
		var lo_q: float = lo_raw if lo_raw < 128 else lo_raw - 256
		var confidence := absf(lo_q) / 127.0

		var wx: float = origin_x + ix * voxel_size + half_vs
		var wy: float = origin_y + iy * voxel_size + half_vs
		var wz: float = z_min + iz * voxel_size + half_vs

		# MuJoCo (x,y,z) -> Godot (x,z,-y), scaled by voxel_size
		var t := Transform3D.IDENTITY.scaled(Vector3(voxel_size, voxel_size, voxel_size))
		t.origin = Vector3(wx, wz, -wy)
		_multi_mesh.set_instance_transform(i, t)

		# Bright white-grey, intensity scales with confidence
		var bright: float = 0.75 + clampf(confidence, 0.0, 1.0) * 0.25
		_multi_mesh.set_instance_color(i, Color(bright * 0.92, bright * 0.94, bright))
