#get stats
import json
import os

BASE_DIR = "data"

def main():
	data_file = open( os.path.join(BASE_DIR, "data.csv"), "r")
	matching_counts = [0,0,0,0,0,0]
	hit_count = 0
	miss_count = 0
	total_count = 0
	first_recommendation_hit = 0
	for line in data_file:
		#2.038:performance,1.349:floating-point,1.344:gcc,1.319:assembly,0.702:pi,0.702:language-agnostic,0.702:algorithm,0.702:unix,0.689:compiler-flags,0.689:fpu,0.689:debugging,0.683:stackoverflow,0.683:sequences,0.683:f#,0.676:tdd,0.676:integration-testing,0.676:automated-tests,0.675:symbols,0.675:profiling,0.675:shared-libraries,0.669:compilation,0.662:java,0.662:random,0.660:x86,0.660:sse,0.659:shadow,0.659:raytracing,0.658:ms-dos,0.658:legacy,;performance algorithm language agnostic unix pi
		ranks =  line.split(";")[0].split(",")[:-2]
		expected_tags = line.split(";")[1].split(" ")
		found_atleast_one = False
		match_count = 0
		for rank in ranks:
			if rank.split(":")[1] in expected_tags:
				if not found_atleast_one:
					hit_count += 1
					found_atleast_one = True
				match_count += 1
		print(str(match_count))
		matching_counts[match_count] += 1
		if not found_atleast_one:
			miss_count += 1
		if ranks[0].split(":")[1] in expected_tags:
			first_recommendation_hit += 1
		total_count += 1

	serialized_dict = {
		"matches": matching_counts,
		"hit_count" : hit_count,
		"miss_count" : miss_count,
		"total_count" : total_count,
		"first_recommendation_hit" : first_recommendation_hit
	}
	with open("stats.json", "w") as f:
		f.write(json.dumps(serialized_dict))


if __name__ == '__main__':
	main()