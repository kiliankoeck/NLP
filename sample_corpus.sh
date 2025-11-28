mkdir -p data/test_set

#5 files from each for testing

find data/raw_xmi/Bundesrat -type f -name "*.xmi" | head -n 5 | xargs -I{} cp {} data/test_set/

#not sure why nationalrat data is under filename "xmi" but might need to change this if run again

find data/raw_xmi/xmi -type f -name "*.xmi" | head -n 5 | xargs -I{} cp {} data/test_set/
