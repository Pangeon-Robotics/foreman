extends Camera3D
## 3D orbit camera like MuJoCo viewer.
##
## Default: perspective orbit, 45 deg down, following robot.
## Mouse drag to rotate, scroll to zoom, middle-click to pan.
## Press 'V' to toggle top-down orthographic.

var _follow_target := Vector3.ZERO
var _top_down := false
var _orbit_yaw := 0.0
var _orbit_pitch := -PI / 4.0  # 45 deg down
var _orbit_distance := 12.0
var _ortho_size := 20.0
var _dragging := false
var _panning := false
var _last_mouse := Vector2.ZERO
var _pan_offset := Vector3.ZERO

const ORBIT_SENSITIVITY := 0.005
const PAN_SENSITIVITY := 0.02
const ZOOM_STEP := 1.0
const ZOOM_STEP_ORTHO := 2.0
const MIN_ORTHO := 5.0
const MAX_ORTHO := 60.0
const MIN_DIST := 3.0
const MAX_DIST := 50.0
const FOLLOW_SMOOTH := 0.08


func _ready() -> void:
	# Start in 3D perspective orbit
	projection = Camera3D.PROJECTION_PERSPECTIVE
	fov = 60.0


func set_follow_target(pos: Vector3) -> void:
	_follow_target = pos


func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_V:
			_top_down = not _top_down
			if _top_down:
				projection = Camera3D.PROJECTION_ORTHOGONAL
				size = _ortho_size
			else:
				projection = Camera3D.PROJECTION_PERSPECTIVE
				fov = 60.0
		elif event.keycode == KEY_R:
			# Reset view
			_pan_offset = Vector3.ZERO
			_orbit_yaw = 0.0
			_orbit_pitch = -PI / 4.0
			_orbit_distance = 12.0

	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			_dragging = event.pressed
			_last_mouse = event.position
		elif event.button_index == MOUSE_BUTTON_MIDDLE:
			_panning = event.pressed
			_last_mouse = event.position
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			_dragging = event.pressed
			_last_mouse = event.position
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			if _top_down:
				_ortho_size = maxf(_ortho_size - ZOOM_STEP_ORTHO, MIN_ORTHO)
				size = _ortho_size
			else:
				_orbit_distance = maxf(_orbit_distance - ZOOM_STEP, MIN_DIST)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			if _top_down:
				_ortho_size = minf(_ortho_size + ZOOM_STEP_ORTHO, MAX_ORTHO)
				size = _ortho_size
			else:
				_orbit_distance = minf(_orbit_distance + ZOOM_STEP, MAX_DIST)

	if event is InputEventMouseMotion:
		var delta: Vector2 = event.position - _last_mouse
		_last_mouse = event.position
		if _panning:
			# Pan: shift orbit center
			var right := global_transform.basis.x
			var up := Vector3.UP
			_pan_offset -= right * delta.x * PAN_SENSITIVITY
			_pan_offset += up * delta.y * PAN_SENSITIVITY
		elif _dragging and not _top_down:
			_orbit_yaw -= delta.x * ORBIT_SENSITIVITY
			_orbit_pitch = clampf(
				_orbit_pitch - delta.y * ORBIT_SENSITIVITY,
				-PI / 2.0 + 0.05, -0.05)


func _process(_delta: float) -> void:
	var center := _follow_target + _pan_offset

	if _top_down:
		var target_pos := Vector3(center.x, 15.0, center.z)
		global_position = global_position.lerp(target_pos, FOLLOW_SMOOTH)
		global_rotation = Vector3(-PI / 2.0, 0, 0)
	else:
		# 3D orbit
		var offset := Vector3(
			cos(_orbit_pitch) * sin(_orbit_yaw) * _orbit_distance,
			-sin(_orbit_pitch) * _orbit_distance,
			cos(_orbit_pitch) * cos(_orbit_yaw) * _orbit_distance,
		)
		var target_pos := center + offset
		global_position = global_position.lerp(target_pos, FOLLOW_SMOOTH)
		look_at(center, Vector3.UP)
