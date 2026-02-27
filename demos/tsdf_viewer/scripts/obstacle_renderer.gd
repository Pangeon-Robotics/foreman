extends Node3D
## Renders ground-truth obstacle volumes as semi-transparent red wireframes.
##
## Receives MSG_OBSTACLES (0x05) data: [u16 count][per-obstacle records].
## Each record: [u8 type][3f pos][size...] where type 0=box (3f half-extents),
## type 1=cylinder (2f radius, half_height).
##
## Visual comparison: grey cubes (TSDF) should overlap red wireframes (truth).

var _meshes: Array[MeshInstance3D] = []


func update_obstacles(data: PackedByteArray) -> void:
	# Clear previous
	for m in _meshes:
		m.queue_free()
	_meshes.clear()

	if data.size() < 2:
		return

	var n := data.decode_u16(0)
	var ofs := 2

	for i in n:
		if ofs >= data.size():
			break

		var otype := data[ofs]; ofs += 1
		var px := data.decode_float(ofs); ofs += 4
		var py := data.decode_float(ofs); ofs += 4
		var pz := data.decode_float(ofs); ofs += 4

		var mi := MeshInstance3D.new()

		if otype == 0:  # box
			var hx := data.decode_float(ofs); ofs += 4
			var hy := data.decode_float(ofs); ofs += 4
			var hz := data.decode_float(ofs); ofs += 4
			var box := BoxMesh.new()
			box.size = Vector3(hx * 2, hz * 2, hy * 2)  # MuJoCo Z-up -> Godot Y-up
			mi.mesh = box
		else:  # cylinder
			var radius := data.decode_float(ofs); ofs += 4
			var half_h := data.decode_float(ofs); ofs += 4
			var cyl := CylinderMesh.new()
			cyl.top_radius = radius
			cyl.bottom_radius = radius
			cyl.height = half_h * 2
			mi.mesh = cyl

		# Semi-transparent red wireframe material
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(1.0, 0.2, 0.2, 0.3)
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		mi.material_override = mat

		# MuJoCo Z-up -> Godot Y-up: (x,y,z) -> (x,z,-y)
		mi.position = Vector3(px, pz, -py)

		add_child(mi)
		_meshes.append(mi)
