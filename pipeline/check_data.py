import earthaccess

regions = {
    "houston": (-95.8, 29.4, -95.0, 30.1),
    "frisco": (-96.95, 33.0, -96.65, 33.3),
}
region = "frisco"

earthaccess.login(persist=True)
results = earthaccess.search_data(
    short_name = "OPERA_L3_DISP-S1_V1",
    bounding_box = regions[region],
    temporal = ("2024-01-01", "2025-10-31"),
    count = 200,
)
print(len(results), "granules found")

from collections import defaultdict

by_ref = defaultdict(list)
for r in results:
    parts = r["umm"]["GranuleUR"].split("_")
    by_ref[parts[4], parts[6][:8]].append(parts[7][:8])

for (frame, ref), secs in sorted(by_ref.items()):
    print(frame, ref, len(secs), min(secs), max(secs))

for r in results[:5]:
    print(r["umm"]["GranuleUR"])

# files = earthaccess.download(results[-1], f"data/raw/{region}")
# print(files)

