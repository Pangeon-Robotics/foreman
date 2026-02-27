extends MeshInstance3D
## Ground-plane cost map overlay.
##
## Renders the unified 2D cost grid as a red gradient:
##   Cost 0:       transparent (free space)
##   Cost 1-253:   red gradient (obstacle proximity)
##   Cost 254:     bright red (lethal)
##   Cost 255:     dark grey (unknown / unscanned)
##
## Backward-compatible: detects uint8 cost map (payload=nx*ny) vs
## legacy bit-packed observation map (payload=ceil(nx*ny/8)).
##
## Uses PackedByteArray bulk writes instead of per-pixel set_pixel()
## for ~20x speedup on 200x200 grids.

var _texture: ImageTexture
var _initialized := false
var _last_nx := 0
var _last_ny := 0

# Legacy bit-packed constants (backward compat)
const OBS_R := 38   # 0.15 * 255
const OBS_G := 128  # 0.50 * 255
const OBS_B := 51   # 0.20 * 255
const OBS_A := 89   # 0.35 * 255
const UNK_R := 13   # 0.05 * 255
const UNK_G := 13   # 0.05 * 255
const UNK_B := 20   # 0.08 * 255
const UNK_A := 77   # 0.30 * 255


func _ready() -> void:
	# Create quad mesh at ground level
	var quad := QuadMesh.new()
	quad.size = Vector2(20.0, 20.0)
	quad.orientation = PlaneMesh.FACE_Y
	mesh = quad

	# Position slightly above ground to avoid z-fighting
	position = Vector3(0, 0.01, 0)

	var mat := StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = Color(1, 1, 1, 0.4)
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material_override = mat
	visible = false


func update_observation_map(data: PackedByteArray) -> void:
	# Header: u16x2 (nx, ny), f32x3 (origin_x, origin_y, voxel_size)
	if data.size() < 16:
		return

	var ofs := 0
	var nx: int = data.decode_u16(ofs); ofs += 2
	var ny: int = data.decode_u16(ofs); ofs += 2
	var origin_x := data.decode_float(ofs); ofs += 4
	var origin_y := data.decode_float(ofs); ofs += 4
	var voxel_size := data.decode_float(ofs); ofs += 4

	var n_cells: int = nx * ny
	var payload_size: int = data.size() - ofs

	# Format detection: uint8 cost map (nx*ny bytes) vs bit-packed (ceil(nx*ny/8))
	if payload_size >= n_cells:
		_update_cost_map(data, ofs, nx, ny, n_cells, origin_x, origin_y, voxel_size)
	else:
		_update_bit_packed(data, ofs, nx, ny, n_cells, origin_x, origin_y, voxel_size)


func _update_cost_map(data: PackedByteArray, ofs: int, nx: int, ny: int,
		n_cells: int, origin_x: float, origin_y: float, voxel_size: float) -> void:
	## Render uint8 cost grid as red gradient overlay.
	var pixel_data := PackedByteArray()
	pixel_data.resize(n_cells * 4)

	for i in n_cells:
		var cost: int = data[ofs + i]
		var p: int = i * 4
		if cost == 0:
			# Free space: nearly transparent
			pixel_data[p] = 0
			pixel_data[p + 1] = 0
			pixel_data[p + 2] = 0
			pixel_data[p + 3] = 13  # alpha ~0.05
		elif cost == 255:
			# Unknown: dark grey
			pixel_data[p] = UNK_R
			pixel_data[p + 1] = UNK_G
			pixel_data[p + 2] = UNK_B
			pixel_data[p + 3] = UNK_A
		elif cost == 254:
			# Lethal: bright red
			pixel_data[p] = 255
			pixel_data[p + 1] = 0
			pixel_data[p + 2] = 0
			pixel_data[p + 3] = 179  # alpha 0.70
		else:
			# Gradient: red proportional to cost
			var t: float = float(cost) / 254.0
			pixel_data[p] = int(t * 255.0)      # R
			pixel_data[p + 1] = 0                 # G
			pixel_data[p + 2] = 0                 # B
			pixel_data[p + 3] = int(38.0 + t * 140.0)  # alpha 0.15..0.70

	_apply_image(nx, ny, pixel_data, origin_x, origin_y, voxel_size)


func _update_bit_packed(data: PackedByteArray, ofs: int, nx: int, ny: int,
		n_cells: int, origin_x: float, origin_y: float, voxel_size: float) -> void:
	## Legacy bit-packed observation map (green=observed, grey=unknown).
	var n_bytes: int = ceili(float(n_cells) / 8.0)
	if data.size() < ofs + n_bytes:
		return

	var pixel_data := PackedByteArray()
	pixel_data.resize(n_cells * 4)

	var bit_data := data.slice(ofs, ofs + n_bytes)
	for byte_idx in n_bytes:
		var b: int = bit_data[byte_idx]
		var base_bit: int = byte_idx * 8
		# Process 8 bits per byte (MSB first — numpy packbits order)
		for bit in 8:
			var idx: int = base_bit + (7 - bit)
			if idx >= n_cells:
				break
			var p: int = idx * 4
			if b & (1 << bit):
				pixel_data[p] = OBS_R
				pixel_data[p + 1] = OBS_G
				pixel_data[p + 2] = OBS_B
				pixel_data[p + 3] = OBS_A
			else:
				pixel_data[p] = UNK_R
				pixel_data[p + 1] = UNK_G
				pixel_data[p + 2] = UNK_B
				pixel_data[p + 3] = UNK_A

	_apply_image(nx, ny, pixel_data, origin_x, origin_y, voxel_size)


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
