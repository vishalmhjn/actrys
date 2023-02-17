import overpy
import requests
import os

RAW_NETWORK = os.environ.get("RAW_NETWORK")
bbox = os.environ.get("BBOX")

useragent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
headers = {
    'Connection': 'keep-alive',
    'sec-ch-ua': '"Google Chrome 80"',
    'Accept': '*/*',
    'Sec-Fetch-Dest': 'empty',
    'User-Agent': useragent,
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Origin': 'https://overpass-turbo.eu',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-Mode': 'cors',
    'Referer': 'https://overpass-turbo.eu/',
    'Accept-Language': '',
    'dnt': '1',
}

query = """
	/*
	This has been generated by the overpass-turbo wizard.
	The original search was:
	“highway=*”
	*/
	[out:xml][timeout:25];
	// gather results
	(
	// query part for: “highway=*”
	node["highway"]("""+bbox+""");
	way["highway"]("""+bbox+""");
	relation["highway"]("""+bbox+""");
	);
	// print results
	out body;
	>;
	out skel qt;
	"""
print(query)
data = {
  'data': query
}

response = requests.post('https://overpass-api.de/api/interpreter', headers=headers, data=data)
with open('../data/'+RAW_NETWORK, 'w') as f:
    f.write(response.text)

# api = overpy.Overpass()
# results = api.parse_xml(response.text)