// Requarements:
// Add simple f(t) graph scrlable by t axis,
// main function is avarage accuracy solid line
// additional dash lines are max and min quantils
// on the mouse hover, corresponding poin on accuracy function highlited by circle,
// at this plase appears boxplot

var timeline_plot = new function () {
	var area_width = 800;
	var area_height = 400;

	var exp_cell_width = 40;
	var time_label_margin = 130;

	var animation_speed = 400;

	var margin = 50;

	var DrawTimeline = function (data, target, time_labels, y_label, brief_descriptions, details_links) {
		var place = target.append('div')
			.style('width', area_width + 3*margin)
			.style('height', area_height + 2*margin + time_label_margin)
			.style('overflow', 'hidden');

		var y_axis_label_container = place.append('svg')
			.attr('width', margin)
			.attr('height', area_height+ 2*margin)
			.style('float', 'left')
				.append('text')
					.attr('x', margin/2)
					.attr('y', area_height/2 + margin)
					.attr('dy', '.6rem')
					.attr('transform', 'rotate(-90 '+ (margin/2) + ',' + (area_height/2 + margin) + ')')
					.attr('text-anchor', 'middle')
					.style('font-size', 17)
					.style('font-weight', 'bold')
					.style('font-family', 'suns-serif')
					.html(y_label)


		var y_container = place.append('svg')
			.attr('width', margin)
			.attr('height', area_height + 2*margin)
			.style('float', 'left')
			.append('g')
				.attr('width', 1)
				.attr('height', area_height)
				.attr('transform', 'translate('+ (margin-1) + ',' + margin+ ')');

		var y = d3.scale.linear()
    		.domain([0, 1])
    		.range([area_height, 0]);

    	var yAxis = d3.svg.axis()
    		.scale(y)
    		.orient("left");

    	y_container.append("g")
    		.attr("class", "y axis")
    		.call(yAxis);

		var content_width = data.length * exp_cell_width + 2; // add two empty cells on edges.

		var highlited_idx = -1;

		function RemoveDataCell(target, idx, data) {
			target.select('#box-'+idx).remove();
		}

		function AddDataCell(target, idx,  data) {
			if (!d3.select('#box-'+idx).empty()) {
				return 0;
			}

			var simple_line = d3.svg.line()
				.interpolate('linear')
				.x(function(d){return d[0];})
				.y(function(d){return d[1];});

			var cell = target.append('g')
				.attr('id', 'box-' + idx)
				.attr('height', area_height)
				.attr('width', exp_cell_width)
				.attr('transform', 'translate(' + (idx*exp_cell_width) + ',0)');

			cell.append('rect')
				.attr('class', 'boxplot-box')
				.attr('width', exp_cell_width)
				.attr('height', y(data[idx].botq) - y(data[idx].topq))
				.attr('x', 0)
				.attr('y', y(data[idx].topq))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			cell.append('circle')
				.attr('class', 'boxplot-acc')
				.attr('cx', exp_cell_width/2)
				.attr('cy', y(data[idx].avg))
				.attr('r', 6)
				.attr('stroke', 'blue')
				.attr('stroke-width', 3)
				.attr('fill', 'white')
				.style('z-index', 100);

			cell.append('path')
				.attr('class', 'helper_line')
				.attr('d', simple_line([[exp_cell_width/2, 0],[exp_cell_width/2, area_height]]))
				.attr('stroke', 'black')
				.attr('stroke-width',  0.2)
				.attr('stroke-dasharray', '10,5')
				.attr('fill', 'none');


			cell.append('path')
				.attr('class', 'boxplot-wish-vert-top')
				.attr('d', simple_line([[exp_cell_width/2 , y(data[idx].topq)],[exp_cell_width/2 , y(data[idx].topw)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			cell.append('path')
				.attr('class', 'boxplot-wish-hor-top')
				.attr('d', simple_line([[0 , y(data[idx].topw)],[exp_cell_width, y(data[idx].topw)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			cell.append('path')
				.attr('class', 'boxplot-wish-vert-bot')
				.attr('d', simple_line([[exp_cell_width/2 , y(data[idx].botq)],[exp_cell_width/2 , y(data[idx].loww)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			cell.append('path')
				.attr('class', 'boxplot-wish-hor-bot')
				.attr('d', simple_line([[0 , y(data[idx].loww)],[exp_cell_width, y(data[idx].loww)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			var median = cell.append('path')
				.attr('class', 'boxplot-median')
				.attr('d', simple_line([[0, y(data[idx].med)], [exp_cell_width, y(data[idx].med)]]))
				.attr('stroke', 'red')
				.attr('stroke-width', 1)
				.attr('fill', 'none');


			highlited_idx = idx;
		}

		function UpdateBox(target, idx, data) {
			var simple_line = d3.svg.line()
				.interpolate('linear')
				.x(function(d){return d[0];})
				.y(function(d){return d[1];});

			target.transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('transform', 'translate(' + idx*exp_cell_width + ',0)');
			target.select('.boxplot-box').transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('width', exp_cell_width)
				.attr('height', y(data[idx].botq) - y(data[idx].topq))
				.attr('x', 0)
				.attr('y', y(data[idx].topq))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			target.select('.boxplot-acc').transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('cx', exp_cell_width/2)
				.attr('cy', y(data[idx].avg))
				.attr('r', 6)
				.attr('stroke', 'blue')
				.attr('stroke-width', 3)
				.attr('fill', 'white');

			target.select('.boxplot-wish-vert-top').transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('d', simple_line([[exp_cell_width/2 , y(data[idx].topq)],[exp_cell_width/2 , y(data[idx].topw)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			target.select('.boxplot-wish-hor-top').transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('d', simple_line([[0 , y(data[idx].topw)],[exp_cell_width, y(data[idx].topw)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			target.select('.boxplot-wish-vert-bot').transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('d', simple_line([[exp_cell_width/2 , y(data[idx].botq)],[exp_cell_width/2 , y(data[idx].loww)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			target.select('.boxplot-wish-hor-bot').transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('d', simple_line([[0 , y(data[idx].loww)],[exp_cell_width, y(data[idx].loww)]]))
				.attr('stroke', 'black')
				.attr('stroke-width', 1)
				.attr('fill', 'none');

			target.select('.boxplot-median').transition()
				.duration(animation_speed*Math.abs(idx - highlited_idx))
				.attr('d', simple_line([[0, y(data[idx].med)], [exp_cell_width, y(data[idx].med)]]))
				.attr('stroke', 'red')
				.attr('stroke-width', 1)
				.attr('fill', 'none');
		}

		var main = place.append('div')
			.attr('class', 'timeline')
			.style('float', 'left')
			.style('overflow-y', 'hidden')
			.style('overflow-x', 'scroll')
			.style('height', area_height + 2 * margin + time_label_margin)
			.style('width', area_width + margin);

		var svg = main.append('svg')
			.attr('width', content_width)
			.attr('height', area_height + 2*margin + time_label_margin);
			
		var base = svg.append('g')
			.attr('width', content_width)
			.attr('height', area_height)
			.attr('transform', 'translate(0,' + margin +')');


		var line_grid = d3.svg.line()
			.interpolate('linear')
			.x(function (d) { return d.x;})
			.y(function (d) { return y(d.y);});

		var half = [{'x':0, 'y': 0.5}, {'x':content_width, 'y':0.5}];
		var half_half = [{'x':0, 'y': 0.25}, {'x':content_width, 'y':0.25}];
		var three_half = [{'x':0, 'y': 0.75}, {'x':content_width, 'y':0.75}];
		var full = [{'x':0, 'y': 1}, {'x':content_width, 'y':1}];

		base.append('path')
			.attr('d', line_grid(half))
			.attr('stroke', 'black')
			.attr('stroke-width', 0.2)
			.attr('stroke-dasharray', '10,5')
			.attr('fill', 'none')

		base.append('path')
			.attr('d', line_grid(half_half))
			.attr('stroke', 'black')
			.attr('stroke-width', 0.2)
			.attr('stroke-dasharray', '10,5')
			.attr('fill', 'none')

		base.append('path')
			.attr('d', line_grid(three_half))
			.attr('stroke', 'black')
			.attr('stroke-width', 0.2)
			.attr('stroke-dasharray', '10,5')
			.attr('fill', 'none')

		base.append('path')
			.attr('d', line_grid(full))
			.attr('stroke', 'black')
			.attr('stroke-width', 0.2)
			.attr('stroke-dasharray', '10,5')
			.attr('fill', 'none')

		var x_min = 0;
		var x_max = data.length;

		ticks = [0];
		tick_labels = [' ']
		for (var i = 0; i < x_max; i++) {
			ticks[i+1] = 0.5 + i;
			tick_labels[i+1] = time_labels[i];
		}

		var x = d3.scale.linear()
    		.domain([x_min, x_max])
    		.range([0, content_width]);

		var xAxis = d3.svg.axis()
    		.scale(x)
    		.tickValues(ticks)
    		.tickFormat(function (d,i) { return tick_labels[i]; })
    		.orient("bottom");

    	var x_axis_object = base.append("g")
    		.attr("class", "x axis")
    		.attr("transform", "translate(0," + area_height + ")")
    		.call(xAxis);

    	x_axis_object.selectAll('text')
    		.attr('dy', '0')
    		.attr('dx', '-80')
    		.attr('transform', 'rotate(-90)');

		AddDataCell(base, 0, data);

		var tip = place.append('div')
			.attr('class', 'timeline-tip');

		svg.on("mouseover", function(d) {
			tip.style('visibility', 'visible');		
            tip.transition()		
                .duration(500)
                .delay(1000)		
                .style("opacity", .9);	

        }).on('mousemove', function (d) {
			var idx = ~~(d3.mouse(this)[0] / exp_cell_width);
			if (idx != highlited_idx) {
				UpdateBox(base.select('#box-0'), idx, data);
				highlited_idx = idx;
			}

			tip.style("left", (d3.event.pageX + 30) + "px")		
                .style("top", (d3.event.pageY + 10) + "px");
            tip.html(brief_descriptions[idx]);

		}).on("mouseout", function(d) {		
            tip.transition()		
                .duration(500)
                .style("opacity", 0);	
            tip.style('visibility', 'hidden')
            	.html(" ");
        }).on("click", function(d) {
         	var idx = ~~(d3.mouse(this)[0] / exp_cell_width);
         	window.open(details_links[idx], details_links[idx]);
        });

		var line_acc = d3.svg.line()
			.interpolate('linear')
			.y(function (d,i) {	return y(d.avg);	})
			.x(function (d,i) {   return exp_cell_width/2 + i*exp_cell_width;});

		var line_top_q = d3.svg.line()
			.interpolate('linear')
			.y(function (d,i) {	return y(d.topq);	})
			.x(function (d,i) {   return exp_cell_width/2 + i*exp_cell_width;});

		var line_bot_q = d3.svg.line()
			.interpolate('linear')
			.y(function (d,i) {	return y(d.botq);	})
			.x(function (d,i) {   return exp_cell_width/2 + i*exp_cell_width;});

		var acc_top_q = base.append('path')
			.attr('d', line_top_q(data))
			.attr('stroke', 'black')
			.attr('stroke-width', 0.5)
			.attr('fill', 'none');

		var acc_bot_q = base.append('path')
			.attr('d', line_bot_q(data))
			.attr('stroke', 'black')
			.attr('stroke-width', 0.5)
			.attr('fill', 'none');

		var acc_path = base.append('path')
			.attr('d', line_acc(data))
			.attr('stroke', 'blue')
			.attr('stroke-width', 2)
			.attr('fill', 'none');
	}

	this.plot = DrawTimeline;
	return this;
}