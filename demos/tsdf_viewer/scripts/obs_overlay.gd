extends MeshInstance3D
## Ground-plane cost map overlay with dual-layer blending.
##
## Renders two cost grids as colored overlays:
##   Robot-view (MSG_OBSERVATION_MAP / 0x04): RED channel
##   God-view   (MSG_GOD_VIEW_COSTMAP / 0x06): BLUE channel
##   Overlap (both nonzero): purple (red + blue)
##
##   Cost 0 in both:   transparent (free space)
##   Cost 1-254:       intensity proportional to cost
##   Cost 255:         dark grey (unknown / unscanned)

var _texture: ImageTexture
var _initialized := false
var _last_nx := 0
var _last_ny := 0

# Stored grid data (raw uint8 arrays, image-oriented: already transposed+flipped)
var _robot_data: PackedByteArray  # nx*ny uint8, from robot-view
var _god_data: PackedByteArray    # nx*ny uint8, from god-view
var _robot_nx := 0
var _robot_ny := 0
var _god_nx := 0
var _god_ny := 0

# Spatial metadata (from whichever grid arrived last — they should match)
var _origin_x := 0.0
var _origin_y := 0.0
var _voxel_size := 0.05
var _grid_nx := 0
var _grid_ny := 0


func _ready() -> void:
	var quad := QuadMesh.new()
	quad.size = Vector2(20.0, 20.0)
	quad.orientation = PlaneMesh.FACE_Y
	mesh = quad

	position = Vector3(0, 0.01, 0)

	var mat := StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = Color(1, 1, 1, 0.4)
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material_override = mat
	visible = false


func update_observation_map(data: PackedByteArray) -> void:
	## Receive robot-view costmap (MSG_OBSERVATION_MAP / 0x04).
	if data.size() < 16:
		return

	var ofs := 0
	var nx: int = data.decode_u16(ofs); ofs += 2
	var ny: int = data.decode_u16(ofs); ofs += 2
	var origin_x := data.decode_float(ofs); ofs += 4
	var origin_y := data.decode_float(ofs); ofs += 4
	var voxel_size := data.decode_float(ofs); ofs += 4

	var n_cells: int = nx * ny
	if data.size() < ofs + n_cells:
		return

	_robot_data = data.slice(ofs, ofs + n_cells)
	_robot_nx = nx
	_robot_ny = ny
	_origin_x = origin_x
	_origin_y = origin_y
	_voxel_size = voxel_size
	_grid_nx = nx
	_grid_ny = ny

	_rebuild_combined()


func update_god_view_costmap(data: PackedByteArray) -> void:
	## Receive god-view costmap (MSG_GOD_VIEW_COSTMAP / 0x06).
	if data.size() < 16:
		return

	var ofs := 0
	var nx: int = data.decode_u16(ofs); ofs += 2
	var ny: int = data.decode_u16(ofs); ofs += 2
	var origin_x := data.decode_float(ofs); ofs += 4
	var origin_y := data.decode_float(ofs); ofs += 4
	var voxel_size := data.decode_float(ofs); ofs += 4

	var n_cells: int = nx * ny
	if data.size() < ofs + n_cells:
		return

	_god_data = data.slice(ofs, ofs + n_cells)
	_god_nx = nx
	_god_ny = ny
	_origin_x = origin_x
	_origin_y = origin_y
	_voxel_size = voxel_size
	_grid_nx = nx
	_grid_ny = ny

	_rebuild_combined()


func _rebuild_combined() -> void:
	## Blend robot-view (red) and god-view (blue) into RGBA image.
	var nx := _grid_nx
	var ny := _grid_ny
	if nx == 0 or ny == 0:
		return

	var n_cells: int = nx * ny
	var has_robot := (_robot_data.size() == n_cells)
	var has_god := (_god_data.size() == n_cells)

	if not has_robot and not has_god:
		return

	var pixel_data := PackedByteArray()
	pixel_data.resize(n_cells * 4)

	for i in n_cells:
		var rc: int = _robot_data[i] if has_robot else 0
		var gc: int = _god_data[i] if has_god else 0
		var p: int = i * 4

		# Both unknown (255) -> grey
		var robot_unknown := (rc == 255)
		var god_unknown := (gc == 255)

		if robot_unknown and god_unknown:
			pixel_data[p] = 60
			pixel_data[p + 1] = 60
			pixel_data[p + 2] = 65
			pixel_data[p + 3] = 102
		elif rc == 0 and gc == 0:
			# Both free -> nearly transparent
			pixel_data[p] = 0
			pixel_data[p + 1] = 0
			pixel_data[p + 2] = 0
			pixel_data[p + 3] = 13
		else:
			# Compute red (robot-view) and blue (god-view) intensities
			var r_val: int = 0
			var r_alpha: int = 13
			if not robot_unknown and rc > 0:
				var t: float = float(rc) / 254.0
				r_val = int(t * 255.0)
				r_alpha = int(38.0 + t * 140.0)
			elif robot_unknown:
				# Unknown in robot-view only -> grey contribution
				r_val = 60
				r_alpha = 102

			var b_val: int = 0
			var b_alpha: int = 13
			if not god_unknown and gc > 0:
				var t: float = float(gc) / 254.0
				b_val = int(t * 255.0)
				b_alpha = int(38.0 + t * 140.0)
			elif god_unknown:
				b_val = 65
				b_alpha = 102

			pixel_data[p] = r_val      # R
			pixel_data[p + 1] = 0      # G
			pixel_data[p + 2] = b_val  # B
			pixel_data[p + 3] = maxi(r_alpha, b_alpha)

	_apply_image(nx, ny, pixel_data, _origin_x, _origin_y, _voxel_size)


func _apply_image(nx: int, ny: int, pixel_data: PackedByteArray,
		origin_x: float, origin_y: float, voxel_size: float) -> void:
	## Create/update texture and position the overlay quad.
	var img := Image.create_from_data(nx, ny, false, Image.FORMAT_RGBA8, pixel_data)

	if _texture == null or nx != _last_nx or ny != _last_ny:
		_texture = ImageTexture.create_from_image(img)
		var mat: StandardMaterial3D = material_override
		mat.albedo_texture = _texture
		_last_nx = nx
		_last_ny = ny
	else:
		_texture.update(img)

	# Position and scale the overlay to match TSDF world extent
	var extent_x: float = nx * voxel_size
	var extent_y: float = ny * voxel_size
	var center_x: float = origin_x + extent_x / 2.0
	var center_y: float = origin_y + extent_y / 2.0

	# Godot quad: centered. MuJoCo (x,y) -> Godot (x,-z)
	position = Vector3(center_x, 0.01, -center_y)
	var q: QuadMesh = mesh
	q.size = Vector2(extent_x, extent_y)

	visible = true
