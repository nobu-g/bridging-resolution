SRC_JSONL_DIR := /path/to/src//jsonl/dir
TGT_DATA_DIR := /path/to/target/data/dir
TGT_JSONL_DIR := $(TGT_DATA_DIR)/jsonl
ID_DIR := $(TGT_DATA_DIR)/split

TRAIN_ID_FILE := $(ID_DIR)/train.id
VALID_ID_FILE := $(ID_DIR)/valid.id
TEST_ID_FILE := $(ID_DIR)/test.id
SHUFFLED_ID_FILE := $(ID_DIR)/shuffled.id


SHELL := /bin/bash -eu

.PHONY: all train valid test
all: train valid test

train: $(TRAIN_ID_FILE)
	cp $(addprefix $(SRC_JSONL_DIR)/,$(shell cat $< | sd '(^.+)$$' '$$1.jsonl')) ${TGT_JSONL_DIR}/train/

$(TRAIN_ID_FILE): $(SHUFFLED_ID_FILE)
	tail -n +151 $< > $@

valid: $(VALID_ID_FILE)
	cp $(addprefix $(SRC_JSONL_DIR)/,$(shell cat $< | sd '(^.+)$$' '$$1.jsonl')) ${TGT_JSONL_DIR}/valid/

$(VALID_ID_FILE): $(SHUFFLED_ID_FILE)
	head -n 150 $< | tail -n 50 > $@

test: $(TEST_ID_FILE)
	cp $(addprefix $(SRC_JSONL_DIR)/,$(shell cat $< | sd '(^.+)$$' '$$1.jsonl')) ${TGT_JSONL_DIR}/test/

$(TEST_ID_FILE): $(SHUFFLED_ID_FILE)
	head -n 100 $< > $@

$(SHUFFLED_ID_FILE): $(SRC_JSONL_DIR)
	ls $< | tr -d '.jsonl' | shuf > $@
