import * as d3 from 'd3';
import * as chromatic from 'd3-scale-chromatic';

import {cmaps} from '../utils/colors';
import {publish, subscribe} from "../utils";
import {and, or, not, AttrFilter} from '../model';
import {ensure_single} from "../utils/events";

import config from './config';
import Group from './group';
import XAxis from './x_axis';
import template from './details.html';
import './style.css';


let root = null;

let msc = null;
let dims = [];
let partitions = [];
let measure = null;
let selected = null;
let highlight = null;


let initial_cmap = 'RdYlBu';
let color_by = null;
let color_by_opt = 'current';
let colorScale = d3.scaleSequential(chromatic['interpolate'+initial_cmap]);

let pattern = null;

let x_axis = XAxis()
  .on('filter', update_filter);

let sy = d3.scaleLinear().range([config.plot_height, 0]);
let y = d3.local();

let group = Group()
    .y(y)
    .color(pt => colorScale(pt[color_by.name]))
    .on('filter', update_filter);

let pts_filters = and();

// TODO: simplify or break up code

export function setup(el) {
  root = typeof el === 'string' && d3.select(el) || el;
  root.classed('details', true);
  root.html(template);

  root.select('.config').select('#color-by')
    .on('change', function(d) {select_color(this.value);});

  root.select('.config').select('#cmap')
    .on('change', function(d) {
      select_cmap(chromatic['interpolate' + this.value])
    })
    .selectAll('option').data(cmaps)
    .enter()
    .append('option')
    .attr('value', d => d.name)
    .property('selected', d => d.name === initial_cmap)
    .text(d => d.name);

  root.select('#details_show_filtered')
    .property('checked', true)
    .on('change', on_show_filtered);

  root.select('#details_use_canvas')
    .property('checked', true)
    .on('change', on_use_canvas);


  subscribe('data.new', (topic, data) => reset(data));
  subscribe('partition.details', (topic, partition, on) => on ? add(partition) : remove(partition));
  subscribe('partition.highlight', (topic, partition, on) => on_highlight(partition, on));
  subscribe('partition.selected', (topic, partition, on) => on_selected(partition, on));
  subscribe('resample.pts', (topic, pts) => on_resample_pts(pts));
}

function reset(data) {
  partitions = [];
  msc = data;
  measure = msc.measure;
  sy.domain(measure.extent);

  dims = msc.dims.map( dim => ({
    name: dim.name,
    extent: dim.extent,
    filter: AttrFilter(dim.name)
  }));

  pts_filters = and();
  let y_filter = AttrFilter(measure.name);
  pts_filters.add(y_filter);
  group.filter(y_filter);
  for (let dim of dims) {
    pts_filters.add(dim.filter);
  }

  group.dims(msc.dims);
  group.measure(measure);

  show_dims();

  let colors = root.select('.config').select('select').selectAll('option')
    .data(['current'].concat(msc.measures.map(m => m.name)));

  colors.enter()
      .append('option')
    .merge(colors)
      .attr('value', d => d)
      .property('selected', d => d === color_by_opt)
      .text(d => d);
  colors.exit().remove();

  select_color(color_by_opt);
}

function on_highlight(partition, on) {
  highlight = on && partition || null;
  root.select('.groups').selectAll('.group')
    .data(partitions, d => d.id)
    .classed('highlight', d => on && d.id === partition.id)
}

function on_selected(partition, on) {
  selected = on && partition || null;

  root.select('.groups').selectAll('.group')
    .data(partitions, d => d.id)
    .classed('selected', d => selected && d.id === selected.id)
}

function show_dims() {
  let axis = root.select('.dims').selectAll('.dim')
    .data(dims);

  let enter = axis.enter()
    .append('div')
      .attr('class', 'dim');
  enter.append('label');
  enter.call(x_axis.create);

  let update = enter.merge(axis);
  update.select('label').text(d => d.name);
  update.call(x_axis);

  axis.exit().remove();
}

function select_cmap(cmap) {
  colorScale.interpolator(cmap);
  update(partitions, true);
}

function select_color(name) {
  color_by_opt = name;

  color_by = name === 'current' && measure || msc.measure_by_name(name);
  colorScale.domain([color_by.extent[1], color_by.extent[0]]);

  update(partitions, true);
}

function on_show_filtered() {
  group.show_filtered(this.checked);
  update(partitions, true);
}

function on_use_canvas() {
  group.use_canvas(this.checked);
  update(partitions, true);
}

function on_resample_pts(pts) {
  if (!selected) {
    console.log('no selected partition. ignored');
    return;
  }
  let p = partitions.find(pr => pr.id === selected.id);
  p.extra = pts;
  update(partitions, true);
}

function add(partition) {
  let reg_curve = partition.regression_curve;

  partitions.push({
    id: partition.id,
    name: partition.alias,
    p: partition,
    pts: partition.pts,
    line: reg_curve.curve,
    area: reg_curve.curve.map((pt, i) => ({pt, std: reg_curve.std[i]}))
  });

  update(partitions);
}

function remove(partition){
  let idx = partitions.findIndex(p => p.id === partition.id);
  if (idx !== -1) {
    partitions.splice(idx, 1);
  }

  update(partitions);
}

function update_filter(attr) {
  for (let pt of msc.pts) {
    pt.filtered = !pts_filters(pt);
  }
  update(partitions, true);
  publish('data.updated');
}

function update(list, all=false) {
  list.sort( (a,b) => a.id - b.id );
  list.forEach( (d, i) => d.x = i);

  root.select('.groups')
    .each( function() {
      y.set(this, pt => sy(pt[measure.name]));
    });

  let t0 = performance.now();

  let groups = root.select('.groups').selectAll('.group')
    .data(list, d => d.id);

  let g = groups.enter()
    .append('div')
      .on('mouseenter', d => publish('partition.highlight', d.p, true))
      .on('mouseleave', d => publish('partition.highlight', d.p, false))
      .call(group.create);

  g.select('.group-header')
    .on('click', ensure_single(d => publish('partition.details', d.p, false)))
    .on('dblclick', d => publish('partition.selected', d.p, d.p !== selected));

  g.merge(groups)
      .classed('highlight', d => highlight && d.id === highlight.id)
      .classed('selected', d => selected && d.id === selected.id)
      .call(group, all);

  groups.exit().call(group.remove);
  let t1 = performance.now();
  console.log(`details update: ${Math.round(t1-t0)} msec`);
}

function select(d) {
  publish('partition.selected', d.p, !selected || d.p.id !== selected.id);
}