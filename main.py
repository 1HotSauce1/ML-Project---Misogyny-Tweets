import convert_text as ct

test_file = 'train.csv'
raw_data = ct.read_data(test_file)
labels = ct.get_labels(raw_data)
string_list = ct.filter_data(raw_data)
print(string_list)
union = ct.lists_union(string_list)
print(len(union))
dictionare = ct.dictionary(string_list, union)

# 12882 words > 2
# 12540 words > 3
