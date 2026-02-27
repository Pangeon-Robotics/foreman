extends MeshInstance3D
## Ground-plane observation overlay.
##
## Shows which areas of the TSDF have been scanned (green)
## vs unknown (dark). Renders as a semi-transparent quad at y=0.01.
##
## Uses PackedByteArray bulk writes instead of per-pixel set_pixel()
## for ~20x speedup on 200x200 grids.

var _texture: ImageTexture
var _initialized := false
var _last_nx := 0
var _last_ny := 0

# Pre-computed RGBA8 byte values (avoids per-pixel Color creation)
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

	var n_bits: int = nx * ny
	var n_bytes: int = ceili(float(n_bits) / 8.0)
	if data.size() < ofs + n_bytes:
		return

	# Build RGBA8 pixel buffer directly (no per-pixel set_pixel calls)
	var pixel_data := PackedByteArray()
	pixel_data.resize(n_bits * 4)

	var bit_data := data.slice(ofs, ofs + n_bytes)
	for byte_idx in n_bytes:
		var b: int = bit_data[byte_idx]
		var base_bit: int = byte_idx * 8
		# Process 8 bits per byte (MSB first — numpy packbits order)
		for bit in 8:
			var idx: int = base_bit + (7 - bit)
			if idx >= n_bits:
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
