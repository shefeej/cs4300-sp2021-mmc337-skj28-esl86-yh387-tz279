from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "MyMuseum"
net_id = "Madeleine Chang: mmc337, Shefali Janorkar: skj28, Esther Lee: esl86, Yvette Hung: yh387, Tiffany Zhong: tz279"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



