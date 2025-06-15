[⚠️ Suspicious Content] # Makefile for Viola-Jones Face Detection using CUDA and OpenCV

# Compiler and flags
NVCC := nvcc
CXXFLAGS := -std=c++14
DEPRECATION_WARNINGS := -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS
ARCH_FLAGS := -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75

# OpenCV (auto-detect flags)
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4)

# Project includes
INCLUDES := -I./include -I../Common -I../Common/include -I../Common/UtilNPP -I/usr/include/opencv4

# CUDA and FreeImage libs
LIBS := -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage

# Output binary name
TARGET := violaJonesFaceDetectionNPP

# Source file
SRC := violaJonesFaceDetectionNPP.cpp

# Clean all intermediates
.PHONY: clean
clean:
	rm -rf ./logs ./output ./*.o ./*.h.gch $(TARGET)

# Default build target
.PHONY: build
build: $(TARGET)

# GPU build rule
$(TARGET): $(SRC)
	@echo "Building violaJonesFaceDetectionNPP.cpp..."
	$(NVCC) $(CXXFLAGS) $(ARCH_FLAGS) -Xcompiler "$(DEPRECATION_WARNINGS)" \
	$(INCLUDES) $(LIBS) $(OPENCV_FLAGS) $^ -o violaJonesFaceDetectionNPP
	mkdir -p ./logs
	mkdir -p ./output

test_sample:
	@echo "Testing GPU runs on sample image..."
	@echo "Logs are stored in ./logs/"
	./violaJonesFaceDetectionNPP -i ./LennaTestImage.pgm -o ./output/LennaTestImage_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/LennaTestImage.log

test_photography:
	mkdir -p ./logs/photography
	mkdir -p ./output/photography
	@echo "Testing runs on images from photography in ./data/photography/..."
	@echo "Logs are stored in ./logs/photography/"
	./violaJonesFaceDetectionNPP -i ./data/photography/BenHur.pgm -o ./output/photography/BenHur_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/BenHur.log
	./violaJonesFaceDetectionNPP -i ./data/photography/ChroniclesOfNarnia.pgm -o ./output/photography/ChroniclesOfNarnia_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/ChroniclesOfNarnia.log
	./violaJonesFaceDetectionNPP -i ./data/photography/GodzillaMinusOne.pgm -o ./output/photography/GodzillaMinusOne_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/GodzillaMinusOne.log
	./violaJonesFaceDetectionNPP -i ./data/photography/HungerGames.pgm -o ./output/photography/HungerGames_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/HungerGames.log
	./violaJonesFaceDetectionNPP -i ./data/photography/JurassicPark.pgm -o ./output/photography/JurassicPark_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/JurassicPark.log
	./violaJonesFaceDetectionNPP -i ./data/photography/LordOfTheRings.pgm -o ./output/photography/LordOfTheRings_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/LordOfTheRings.log
	./violaJonesFaceDetectionNPP -i ./data/photography/StarWars.pgm -o ./output/photography/StarWars_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/StarWars.log
	./violaJonesFaceDetectionNPP -i ./data/photography/ThePrincessBride.pgm -o ./output/photography/ThePrincessBride_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/photography/ThePrincessBride.log

test_anime:
	mkdir -p ./logs/anime
	mkdir -p ./output/anime
	@echo "Testing runs on images from anime in ./data/anime/..."
	@echo "Logs are stored in ./logs/anime/"
	./violaJonesFaceDetectionNPP -i ./data/anime/Bleach.pgm -o ./output/anime/Bleach_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/Bleach.log
	./violaJonesFaceDetectionNPP -i ./data/anime/DemonSlayer.pgm -o ./output/anime/DemonSlayer_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/DemonSlayer.log
	./violaJonesFaceDetectionNPP -i ./data/anime/DungeonMeshi.pgm -o ./output/anime/DungeonMeshi_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/DungeonMeshi.log
	./violaJonesFaceDetectionNPP -i ./data/anime/FullmetalAlchemist.pgm -o ./output/anime/FullmetalAlchemist_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/FullmetalAlchemist.log
	./violaJonesFaceDetectionNPP -i ./data/anime/MyHeroAcademia.pgm -o ./output/anime/MyHeroAcademia_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/MyHeroAcademia.log
	./violaJonesFaceDetectionNPP -i ./data/anime/Naruto.pgm -o ./output/anime/Naruto_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/Naruto.log
	./violaJonesFaceDetectionNPP -i ./data/anime/Pokemon.pgm -o ./output/anime/Pokemon_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/Pokemon.log
	./violaJonesFaceDetectionNPP -i ./data/anime/SpyXFamily.pgm -o ./output/anime/SpyXFamily_detected.pgm -f ./haarcascade_frontalface_default.xml > ./logs/anime/SpyXFamily.log

perf:
	@echo "Parsing logs..."
	@echo "Plots generated in logs/plots/"
	python3 run_parser.py ./logs ./logs

clean:
	rm -f violaJonesFaceDetectionNPP
	$(RM) -rf ./logs ./output ./*.o ./*.h.gch