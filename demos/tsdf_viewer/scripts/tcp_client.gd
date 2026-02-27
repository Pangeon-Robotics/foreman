extends Node
## TCP client with length-prefixed binary framing.
##
## Protocol: [u32 total_len][u8 msg_type][payload], little-endian.
## Emits typed signals for each message type.

signal connected
signal disconnected
signal robot_state_received(data: PackedByteArray)
signal tsdf_received(data: PackedByteArray)
signal path_received(data: PackedByteArray)
signal observation_map_received(data: PackedByteArray)
signal obstacles_received(data: PackedByteArray)

const MSG_ROBOT_STATE := 0x01
const MSG_TSDF_OCCUPIED := 0x02
const MSG_ASTAR_PATH := 0x03
const MSG_OBSERVATION_MAP := 0x04
const MSG_OBSTACLES := 0x05

var _tcp := StreamPeerTCP.new()
var _buf := PackedByteArray()
var _connected := false

@export var host := "127.0.0.1"
@export var port := 9877
@export var auto_reconnect := true

var _reconnect_timer := 0.0
const RECONNECT_INTERVAL := 2.0


func _ready() -> void:
	_try_connect()


func _try_connect() -> void:
	_tcp.connect_to_host(host, port)


func _process(delta: float) -> void:
	_tcp.poll()
	var status := _tcp.get_status()

	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not _connected:
			_connected = true
			_reconnect_timer = 0.0
			connected.emit()

		# Read available data
		var avail := _tcp.get_available_bytes()
		if avail > 0:
			var chunk := _tcp.get_data(avail)
			if chunk[0] == OK:
				_buf.append_array(chunk[1])
		_parse_messages()

	elif status == StreamPeerTCP.STATUS_NONE or status == StreamPeerTCP.STATUS_ERROR:
		if _connected:
			_connected = false
			_buf.clear()
			disconnected.emit()
		if auto_reconnect:
			_reconnect_timer += delta
			if _reconnect_timer >= RECONNECT_INTERVAL:
				_reconnect_timer = 0.0
				_tcp = StreamPeerTCP.new()
				_try_connect()


func _parse_messages() -> void:
	# Parse all complete messages in buffer
	while _buf.size() >= 5:  # minimum: 4 (len) + 1 (type)
		var total_len := _buf.decode_u32(0)
		if total_len < 1 or total_len > 1_000_000:
			# Invalid frame — discard buffer
			_buf.clear()
			return
		var frame_size: int = 4 + total_len
		if _buf.size() < frame_size:
			return  # incomplete message

		var msg_type := _buf[4]
		var payload := _buf.slice(5, frame_size)

		# Dispatch
		match msg_type:
			MSG_ROBOT_STATE:
				robot_state_received.emit(payload)
			MSG_TSDF_OCCUPIED:
				tsdf_received.emit(payload)
			MSG_ASTAR_PATH:
				path_received.emit(payload)
			MSG_OBSERVATION_MAP:
				observation_map_received.emit(payload)
			MSG_OBSTACLES:
				obstacles_received.emit(payload)

		# Consume this message
		_buf = _buf.slice(frame_size)


func is_connected_to_server() -> bool:
	return _connected
