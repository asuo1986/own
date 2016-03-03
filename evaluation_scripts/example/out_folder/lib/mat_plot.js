var mat_plot = new function () {
	var mat_width = 800;
	var mat_height = 800;

	var mat_cell_margin = 3;

	var max_cell_width = 100;
	var max_cell_height = 100;

	var min_cell_width = 35;
	var min_cell_height = 35;

	var rows_label_cell_width = 170;
	var cols_label_cell_height = 170;

	function applyMax(arr) { 
		return Math.max.apply(null, arr); 
	};

	function applyMin(arr) {
		return Math.min.apply(null, arr);
	};

	function CellColor(value, data) {
		var max_val = applyMax(data.map(applyMax));
		var min_val = applyMin(data.map(applyMin));

		if (max_val == min_val) {
			return {'ground': 'rgb(255,255,255)', 'text': 'rgb(0,0,0)'};
		} else {
			var base_val = Math.round(255 - (value - min_val) * 150 / (max_val - min_val)).toString();
			var color = 'rgb(' + base_val + ',' + base_val + ', 255)';
			var text_color = (base_val < 140) ? 'rgb(255,255,255)' : 'rgb(0,0,0)';
			return {'ground': color, 'text': text_color};
		}
	};

	function CellSize(base, svg, rows, cols) {
		var cell_width = Math.round((mat_width - (cols + 2) * mat_cell_margin - rows_label_cell_width) / 
			(cols - 1));
		var cell_height = Math.round((mat_height - (rows + 2) * mat_cell_margin - cols_label_cell_height)/ 
			(rows - 1));

		var new_pannel_width = mat_width;
		var new_pannel_height = mat_height;
		if (cell_width < min_cell_width || cell_height < min_cell_height ) {
			base.style('overflow', 'scroll')
		}
		if (cell_width < min_cell_width) {
			cell_width = min_cell_width;
			new_pannel_width = cell_width*(cols - 1) + mat_cell_margin*(cols + 2) + rows_label_cell_width;
			svg.attr('width', new_pannel_width);
		} 
		if (cell_height < min_cell_height) {
			cell_height = min_cell_height;
			new_pannel_height = cell_width*(rows -1) + mat_cell_margin*(rows + 2) + cols_label_cell_height;
			svg.attr('height', new_pannel_height);
		}
		if (cell_width > max_cell_height) {
			cell_width = max_cell_height;
			new_pannel_width = cell_width*(cols - 1) + mat_cell_margin*(cols + 2) + rows_label_cell_width;
			svg.attr('width', new_pannel_width);
			base.style('width', new_pannel_width);
		}
		if (cell_height > max_cell_height) {
			cell_height = max_cell_height;
			new_pannel_height = cell_width*(rows -1) + mat_cell_margin*(rows + 2) + cols_label_cell_height;
			svg.attr('height', new_pannel_height);
			base.style('height', new_pannel_height);
		}

		return {'w': cell_width, 'h': cell_height};
	}

	function AddDataSell(svg, i, j, cell_size, data) {
		var cell = svg.append('g')
			.attr('x', j)
			.attr('y', i)
			.attr('width', cell_size.w)
			.attr('height', cell_size.h)
			.attr('class', 'cell-data')
			.attr('transform', 'translate(' + 
				((j - 1)  * (cell_size.w + mat_cell_margin) + mat_cell_margin + 
				rows_label_cell_width) + ',' + 
				((i - 1) * (cell_size.h + mat_cell_margin) + mat_cell_margin + 
				cols_label_cell_height) + ')');

		var color_palet = CellColor(data[i-1][j-1], data);
		var rect = cell.append('rect')
			.attr('x', 0)
			.attr('y', 0)
			.attr('width', cell_size.w)
			.attr('height', cell_size.h)
			.style('fill', color_palet.ground);
		if (i == j) {
			rect.style('stroke-width', '1').style('stroke', 'black')
		}
		
		cell.append('text')
			.attr('x', Math.round(cell_size.w / 2))
			.attr('y', Math.round(cell_size.h / 2))
			.attr("dy", ".35em")
			.attr('fill', color_palet.text)
			.attr('text-anchor', 'middle')
			.text(data[i-1][j-1].toString());
	};

	function AddHover(base, data, rows_labels, cols_labels) {
		var tip = base.append("div").attr('class', 'mat-tip');

		base.on("mouseover", function(d) {
			tip.style('visibility', 'visible');		
            tip.transition()		
                .duration(500)
                .delay(1000)		
                .style("opacity", .9);		
        }).on("mousemove", function(d) {
            tip.style("left", (d3.event.pageX + 10) + "px")		
                .style("top", (d3.event.pageY + 10) + "px");
           	try {
            	var under = document.elementFromPoint(d3.event.pageX - window.pageXOffset, 
            		d3.event.pageY - window.pageYOffset);
            	if (under.nodeName == 'text' || under.nodeName == 'rect') {
            		var i = under.parentNode.getAttribute('y');
            		var j = under.parentNode.getAttribute('x');
            		tip.html("Acc: " + data[i-1][j-1] + '\n' +
            			"Exp: " + rows_labels[i-1] + '\n' +
            			"Pre: " + cols_labels[j-1] + '\n');
            	}
        	} catch (err) { 
        		tip.style('visibility', 'hidden');
        	};
        }).on("mouseout", function(d) {		
            tip.transition()		
                .duration(500)		
                .style("opacity", 0);	
            tip.style('visibility', 'hidden')
            	.html(" ");
        });
	};

	var DrowMat = function (data, rows_labels, cols_labels, target) {
		var drawing_promise = new Promise(function (resolve, reject) { try {
			var base = target.append("div").attr('class', 'mat');

			AddHover(base, data, rows_labels, cols_labels);

			var svg = base.append("svg")
				.attr('width', mat_width)
				.attr('height', mat_height);

			rows = cols_labels.length + 1;
			cols = rows_labels.length + 1;

			var cell_size = CellSize(base, svg, rows, cols);

			for (var i = 1; i < rows; i++) {
				for (var j = 1; j < cols; j++) {
					AddDataSell(svg, i, j, cell_size, data);
				}
			}
					
			for (var i = 1; i < cols; i++) {
				// Print cols labels
				var cell = svg.append('g')
					.attr('x', i)
					.attr('y', 0)
					.attr('width', cell_size.w)
					.attr('height', cols_label_cell_height)
					.attr('transform', 'translate(' + 
						((i - 1)  * (cell_size.w + mat_cell_margin) + mat_cell_margin + 
						rows_label_cell_width) + ',' + mat_cell_margin + ')');

				cell.append('text')
					.attr('x', Math.round(cell_size.w / 2))
					.attr('y', Math.round(cols_label_cell_height / 2))
					.attr("dy", ".35em")
					.attr('fill', 'rgb(0,0,0)')
					.attr('text-anchor', 'middle')
					.attr('transform', 'rotate(-90 ' + Math.round(cell_size.w / 2) + ',' + Math.round(cols_label_cell_height / 2) + ')')
					.text(cols_labels[i-1]);
			}
						
			for (var i = 1; i < rows; i++) {
				// Print rows labels
				var cell = svg.append('g')
					.attr('x', 0)
					.attr('y', i)
					.attr('width', rows_label_cell_width)
					.attr('height', cell_size.h)
					.attr('transform', 'translate(' + mat_cell_margin + ',' + 
						((i - 1) * (cell_size.h + mat_cell_margin) + mat_cell_margin + 
						cols_label_cell_height) + ')');

				cell.append('text')
					.attr('x', Math.round(rows_label_cell_width / 2))
					.attr('y', Math.round(cell_size.h / 2))
					.attr("dy", ".35em")
					.attr('fill', 'rgb(0,0,0)')
					.attr('text-anchor', 'middle')
					.text(rows_labels[i-1]);
			}

			resolve('Mat ploted');
		} catch (err) {
			reject('Plot failed: ' + err.message);
		}}).then(function (value) {
			console.log(value);
		}, function (value) {
			console.log(value);
		});
	};

	this.plot = DrowMat;
	return this;
}();