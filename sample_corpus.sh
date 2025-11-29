mkdir -p data/test_set

#5 files from each for testing
find data/raw_xmi -type f -name "BR_*.xmi" | head -n 5 | xargs -I{} cp {} data/test_set/
find data/raw_xmi -type f -name "NR_*.xmi" | head -n 5 | xargs -I{} cp {} data/test_set/