import * as d3 from 'd3';


export default function BoxPlot() {
  let margin = {top: 0, right: 0, bottom: 10, left: 50};

  let width = 1,
    height = 1,
    duration = 400,
    domain = null,
    value = Number,
    tickFormat = null;

  // For each small multiple…
  function box(selection) {
    selection
        .each(function (d, i) {
          let svg = d3.select(this);
          let
            n = d.n,
            min = d.min,
            max = d.max;


          // // Compute outliers. If no whiskers are specified, all data are "outliers".
          // // We compute the outliers as indices, so that we can join across transitions!
          // let outlierIndices = whiskerIndices
          //   ? d3.range(0, whiskerIndices[0]).concat(d3.range(whiskerIndices[1] + 1, n))
          //   : d3.range(n);

          svg.selectAll('.frame').data([1])
            .enter()
            .append('rect')
            .attr('class', 'frame')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);

          let x1 = d3.scaleLinear()
            .domain(d.extent)
            .range([0, width]);

          let x0 = this.__chart__ || d3.scaleLinear()
            .domain([0, Infinity])
            .range(x1.range());

          // Stash the new scale.
          this.__chart__ = x1;

          // Note: the box, median, and box tick elements are fixed in number,
          // so we only have to handle enter and update. In contrast, the outliers
          // and other elements are letiable, so we need to exit them! letiable
          // elements also fade in and out.

          // Update center line: the vertical line spanning the whiskers.
          let center = svg.selectAll("line.center")
            .data([[min, max]]);

          center.enter().insert("line", "rect")
            .attr("class", "center")
            .attr("y1", height / 2)
            .attr("x1", d => x0(d[0]))
            .attr("y2", height / 2)
            .attr("x2", d => x0(d[1]))
            .style("opacity", 1e-6)
            .transition()
            .duration(duration)
            .style("opacity", 1)
            .attr("x1", d => x1(d[0]))
            .attr("x2", d => x1(d[1]));

          center.transition()
            .duration(duration)
            .style("opacity", 1)
            .attr("x1", function (d) {
              return x1(d[0]);
            })
            .attr("x2", function (d) {
              return x1(d[1]);
            });

          center.exit().transition()
            .duration(duration)
            .style("opacity", 1e-6)
            .attr("x1", function (d) {
              return x1(d[0]);
            })
            .attr("x2", function (d) {
              return x1(d[1]);
            })
            .remove();

          // Update innerquartile box.
          let box = svg.selectAll("rect.box")
            .data([d.quantile]);

          box.enter().append("rect")
            .attr("class", "box")
            .attr("y", 0)
            .attr("x", function (d) {
              return x0(d[0]);
            })
            .attr("width", d => x0(d[2]) - x0(d[0]))
            .attr("height", height)
            .transition()
            .duration(duration)
            .attr("x", function (d) {
              return x1(d[0]);
            })
            .attr("width", d => x1(d[2]) - x1(d[0]));

          box.transition()
            .duration(duration)
            .attr("x", d => x1(d[0]))
            .attr("width", d => x1(d[2]) - x1(d[0]));

          // Update median line.
          let medianLine = svg.selectAll("line.median")
            .data([d.quantile[1]]);

          medianLine.enter().append("line")
            .attr("class", "median")
            .attr("y1", x0)
            .attr("x1", 0)
            .attr("y2", height)
            .attr("x2", x0)
            .transition()
            .duration(duration)
            .attr("x1", x1)
            .attr("x2", x1);

          medianLine.transition()
            .duration(duration)
            .attr("x1", x1)
            .attr("x2", x1);

          // Update whiskers.
          let whisker = svg.selectAll("line.whisker")
            .data([min, max]);

          whisker.enter().insert("line", "circle, text")
            .attr("class", "whisker")
            .attr("y1", 0)
            .attr("x1", x0)
            .attr("y2", height)
            .attr("x2", x0)
            .style("opacity", 1e-6)
            .transition()
            .duration(duration)
            .attr("x1", x1)
            .attr("x2", x1)
            .style("opacity", 1);

          whisker.transition()
            .duration(duration)
            .attr("x1", x1)
            .attr("x2", x1)
            .style("opacity", 1);

          whisker.exit().transition()
            .duration(duration)
            .attr("x1", x1)
            .attr("x2", x1)
            .style("opacity", 1e-6)
            .remove();

          // Update outliers.
          // let outlier = svg.selectAll("circle.outlier")
          //   .data(outlierIndices, Number);

          // outlier.enter().insert("circle", "text")
          //   .attr("class", "outlier")
          //   .attr("r", 5)
          //   .attr("cx", width / 2)
          //   .attr("cy", function(i) { return x0(d[i]); })
          //   .style("opacity", 1e-6)
          //   .transition()
          //   .duration(duration)
          //   .attr("cy", function(i) { return x1(d[i]); })
          //   .style("opacity", 1);
          //
          // outlier.transition()
          //   .duration(duration)
          //   .attr("cy", function(i) { return x1(d[i]); })
          //   .style("opacity", 1);
          //
          // outlier.exit().transition()
          //   .duration(duration)
          //   .attr("cy", function(i) { return x1(d[i]); })
          //   .style("opacity", 1e-6)
          //   .remove();

          // Compute the tick format.
          let format = tickFormat || x1.tickFormat(8);

          // Update box ticks.
          // let boxTick = svg.selectAll("text.box")
          //   .data(d.quantile);
          //
          // boxTick.enter().append("text")
          //   .attr("class", "box")
          //   .attr("dx", ".3em")
          //   .attr("dy", function(d, i) { return i & 1 ? 6 : -6 })
          //   .attr("y", function(d, i) { return i & 1 ? width : 0 })
          //   .attr("x", x0)
          //   .attr("text-anchor", function(d, i) { return i & 1 ? "start" : "end"; })
          //   .text(format)
          //   .transition()
          //   .duration(duration)
          //   .attr("x", x1);
          //
          // boxTick.transition()
          //   .duration(duration)
          //   .text(format)
          //   .attr("x", x1);

          // Update whisker ticks. These are handled separately from the box
          // ticks because they may or may not exist, and we want don't want
          // to join box ticks pre-transition with whisker ticks post-.
          let whiskerTick = svg.selectAll("text.whisker")
            .data([min, max]);

          whiskerTick.enter().append("text")
            .attr("class", "whisker")
            .attr("dx", "-1em")
            .attr("dy", 10)
            .attr("y", height)
            .attr("x", x0)
            .text(format)
            .style("opacity", 1e-6)
            .transition()
            .duration(duration)
            .attr("x", x1)
            .style("opacity", 1);

          whiskerTick.transition()
            .duration(duration)
            .text(format)
            .attr("x", x1)
            .style("opacity", 1);

          whiskerTick.exit().transition()
            .duration(duration)
            .attr("x", x1)
            .style("opacity", 1e-6)
            .remove();
        });
    d3.timerFlush();
  }


  box.width = function(x) {
    if (!arguments.length) return width;
    width = x;
    return box;
  };

  box.height = function(x) {
    if (!arguments.length) return height;
    height = x;
    return box;
  };

  box.tickFormat = function(x) {
    if (!arguments.length) return tickFormat;
    tickFormat = x;
    return box;
  };

  box.duration = function(x) {
    if (!arguments.length) return duration;
    duration = x;
    return box;
  };

  box.domain = function(x) {
    if (!arguments.length) return domain;
    domain = x == null ? x : d3.functor(x);
    return box;
  };

  box.value = function(x) {
    if (!arguments.length) return value;
    value = x;
    return box;
  };

  box.whiskers = function(x) {
    if (!arguments.length) return whiskers;
    whiskers = x;
    return box;
  };

  box.quartiles = function(x) {
    if (!arguments.length) return quartiles;
    quartiles = x;
    return box;
  };

  return box;
};



