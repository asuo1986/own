var table_plot = new function () {
	var DrawTable = function (data, target) {
		var table = target.append('table')
			.attr('class', 'details-table');

		for (var i = 0; i < data.length; i++) {
			var row = table.append('tr');
			row.append('td')
				.html(data[i][0]);
			row.append('td')
				.html(data[i][1]);
		}
	}

	this.plot = DrawTable;
	return this;
}();