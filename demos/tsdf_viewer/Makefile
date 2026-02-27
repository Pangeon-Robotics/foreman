GODOT_VERSION := 4.4.1
GODOT_DIR := $(HOME)/.local/share/godot
GODOT_BIN := $(GODOT_DIR)/Godot_v$(GODOT_VERSION)-stable_linux.x86_64
GODOT_URL := https://github.com/godotengine/godot/releases/download/$(GODOT_VERSION)-stable/Godot_v$(GODOT_VERSION)-stable_linux.x86_64.zip

.PHONY: run install clean

run: $(GODOT_BIN)
	$(GODOT_BIN) --path $(CURDIR) &

install: $(GODOT_BIN)
	@echo "Godot $(GODOT_VERSION) ready at $(GODOT_BIN)"

$(GODOT_BIN):
	@echo "Downloading Godot $(GODOT_VERSION)..."
	mkdir -p $(GODOT_DIR)
	curl -L -o /tmp/godot.zip $(GODOT_URL)
	unzip -o /tmp/godot.zip -d $(GODOT_DIR)
	chmod +x $(GODOT_BIN)
	rm /tmp/godot.zip
	@echo "Installed: $(GODOT_BIN)"

clean:
	rm -rf .godot
