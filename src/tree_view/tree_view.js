import * as d3 from 'd3';
import * as chromatic from "d3-scale-chromatic";

import {publish, subscribe} from "../utils/pubsub";
import {AttrRangeFilter, RangeAttrRangeFilter} from "../model/attr_filter";
import {and} from '../model/filter';
import Tree from './lifeline';
import Slider from '../components/slider'

import template from './tree_view.html';
import feature_template from './feature.html';
import './style.css';
import './feature.css';

let root = null;
let msc = null;
let tree = Tree();

let format= d3.format('.3f');

let slider = Slider();
let feature_slider = Slider();
let prevent = false;
let saved = [0, 0];

let features = [];
let version='1';

init_features();

function init_features() {
  add_fitness_feature();
  add_parent_feature();
  add_sibling_feature();
  add_minmax_feature()
}

function add_fitness_feature() {
  let name = 'fitness';
  let domain = [0.8, 1];
  let cmap = ["#3e926e", "#f2f2f2", "#9271e2"];
  let colorScale = d3.scaleSequential(d3.interpolateRgbBasis(cmap)).domain(domain);

  features.push({
    id: 0, name: name, label: 'fitness',
    domain: domain,
    cmp: (a, b) => a > b,
    filter2: AttrRangeFilter('fitness', null),
    active: false,
    cmap: cmap,
    colorScale: colorScale,
    color: p => colorScale(p.model[name]),
    ticks:{n: 4, format:'.2f'},
    interface: true
  });
}

function add_parent_feature() {
  let name = 'parent_similarity';
  let domain = [-1, 1];
  let cmap = ["#4472a5", "#f2f2f2", "#d73c4a"];
  let colorScale = d3.scaleSequential(d3.interpolateRgbBasis(cmap)).domain(domain);

  features.push({
    id: 1, name: name, label: 'parent similarity',
    domain: domain,
    cmp: (a, b) => a < b,
    filter2: AttrRangeFilter('parent_similarity', domain),
    active: false,
    cmap: cmap,
    colorScale: colorScale,
    color: p => {
      let c = colorScale(p.model[name]);
      return c;
    },
    ticks:{n: 4, format:'.2f'},
    interface: true
  });
}

function add_sibling_feature() {
  let name = 'sibling_similarity';
  let domain = [-1, 1];
  let cmap = ["#4472a5", "#f2f2f2", "#d73c4a"];
  let colorScale = d3.scaleSequential(d3.interpolateRgbBasis(cmap)).domain(domain);

  features.push({
    id: 2, name: name, label: 'sibling similarity',
    domain: domain,
    cmp: (a, b) => a < b,
    filter2: AttrRangeFilter('sibling', domain),
    active: false,
    cmap: cmap,
    colorScale: colorScale,
    color: p => colorScale(p.model[name]),
    ticks:{n: 4, format:'.2f'},
    interface: true
  });
}

function add_minmax_feature() {
  let name = 'minmax';
  let domain = [0, 1];
  let cmap = chromatic['interpolateRdYlBu'];
  // let colorScale = d3.scaleSequential(cmap).domain(domain);
  let colorScale = d3.scaleLinear().domain(domain).range(['#bce2fe', /*'#fffebe',*/ '#fd666e']);

  features.push({
    id: 3, name: name, label: 'min max',
    domain: domain,
    cmp: (a, b) => a < b,
    filter2: RangeAttrRangeFilter('minmax', domain),
    active: false,
    cmap: cmap,
    colorScale: colorScale,
    color: p => {
      let c= colorScale(p);
      return c;
    },
    ticks:{n: 4, format:'.2f'},
    interface: true
  });
}


let sliders = [
  { id: 'x', type: 'linear', domain: [Number.EPSILON, 1], ticks: {n: 5, format: 'd'}, selection: [0, 1]},
  { id: 'y', type: 'log', domain: [Number.EPSILON, 1], ticks:{ n: 4, format: '.1e'}, selection: [0.3, 1]}
];

let filter = and();

export function setup(el) {
  root = d3.select(el);
  root.html(template);

  load_setup();

  sliders.forEach(slider => {
    slider.type = localStorage.getItem(`tree.${slider.id}.type`);
    let s = localStorage.getItem(`tree.${slider.id}.selection`);
    console.log('slider section', slider.id, s);
    slider.selection = s && JSON.parse(s) || slider.domain;
  });

  root.select('#tree-x-type')
    .on('change', select_x_type)
    .property('value', sliders[0].type);

  root.select('#tree-y-type')
    .on('change', select_y_type)
    .property('value', sliders[1].type);

  tree
    .on('highlight', (node, on) => publish('partition.highlight', node, on))
    .on('select', (node, on) => publish('partition.selected', node, on))
    .on('details', (node, on) => publish('partition.details', node, on))
    .x_type(sliders[0].type)
    .y_type(sliders[1].type)
    .filter(filter);

  root.select('.tree').call(tree);
  resize();

  slider.on('change', on_slider_change);

  let s = root.selectAll('.slider')
    .data([sliders[1], sliders[0]])
    .call(slider);

  features.forEach(f => filter.add(f.filter2));

  subscribe('init', init);
  subscribe('data.new', (topic, data) => reset(data));
  subscribe('data.loaded', (topic, data) => reset(null));
  subscribe('data.updated', () => tree.update());

  subscribe('partition.highlight', (topic, partition, on) => tree.highlight(partition, on));
  subscribe('partition.details', (topic, partition, on) => tree.details(partition, on));
  subscribe('partition.selected', (topic, partition, on) => tree.selected(partition, on));
   // subscribe('persistence.range', (topic, range) => set_persistence_range(range) );
}

function init() {
  let d3features = d3.select('.filtering_view')
    .selectAll('.feature')
    .data(features.filter(f => f.interface))
    .enter()
    .append('div')
    .html(feature_template);

  d3features.select('.feature-name').text(d => d.label);

  d3features.select('.feature-active')
    .property('checked', d => d.active)
    .on('change', activate_filter);

  d3features.select('.feature-slider2')
    .call(feature_slider);

  feature_slider.on('change', update_feature);

  d3features.select('.feature-cmap')
    .style('background-image', d => `linear-gradient(to right, ${d.cmap.join()}`);

  let idx = +localStorage.getItem('feature.color_by') || 0;
  tree.color_by(features[idx]);

  d3.select('.filtering_view .feature-color')
    .on('change', update_color_by)
    .selectAll('option')
    .data(features)
    .enter()
    .append('option')
    .attr('value', d => d.id)
    .property('selected', d => +d.id === idx)
    .text(d => d.label);

  let show = localStorage.getItem('feature.show_opt');
  d3.select('.filtering_view').selectAll('input[name="show-nodes')
    .property('checked', function() { return this.value === show;})
    .on('change', function() {
      tree.show(this.value);
      localStorage.setItem('feature.show_opt', this.value);
    });
}


export function set_size(w, h) {
  if (root) resize();
}

function load_setup() {
  if (localStorage.getItem('tree_view.version') === version) {
    features.forEach(f => {
      let selection = localStorage.getItem(`feature.${f.name}.selection`);
      f.selection = selection && JSON.parse(selection) || f.domain;
      f.active = localStorage.getItem(`feature.${features[0].name}.active`) === 'on';
      f.filter2.active(f.active);
      f.filter2.range(f.selection);
    });
  } else {
    localStorage.setItem('tree_view.version', version);
  }
}

function resize() {
  let rw = parseInt(root.style('width'));
  let rh = parseInt(root.style('height'));
  let ch = parseInt(root.select('.config').style('height'));

  tree.set_size(rw, rh - ch);
}


function reset(data) {
  msc = data;

  process_data();

  if (!data)
    tree.data([], null);
  else {
    tree.x_range([0, msc.pts.length]);
    sliders[0].domain = [Number.EPSILON, msc.pts.length];
    root.selectAll('.slider').call(slider);

    let mmf = features.find(f => f.name === 'minmax');
    mmf.domain =  [msc.minmax[0], msc.minmax[1]];
    mmf.selection = mmf.domain.concat();
    mmf.colorScale.domain(mmf.domain);
    mmf.filter2.range(mmf.domain);

    console.log(`new data: min/max: ${mmf.domain}`);

    d3.selectAll('.feature-slider2')
      .call(feature_slider);

    tree.data(msc.partitions, msc.tree);
  }
}

function process_data() {
  if (!msc) return;
  visit(msc.tree, features[1].name, node => node.parent);
  visit(msc.tree, features[2].name, sibling );
  msc.partitions.forEach( p => p.model.minmax = p.minmax);


  function visit(node, feature, func) {
    if (!node ) {
      console.log("**** process_data: null node");
      return;
    }
    let other = func(node);
    if (other) {
      let c = node.model.linear_reg.coeff;
      let o = other.model.linear_reg.coeff;

      if (c.norm === undefined) c.norm = norm(c);
      if (o.norm === undefined) o.norm = norm(o);

      node.model[feature] = dot(c,o)/(c.norm * o.norm);
    } else {
      node.model[feature] = 1;
    }
    for (let child of node.children)
      visit(child, feature, func);
  }

  function norm(vec) {
    return Math.sqrt(vec.reduce( (a,v) => a + v*v, 0));
  }

  function dot(v1, v2) {
    let d = 0;
    for (let i=0; i<v1.length; i++) d += v1[i]*v2[i];
    return d;
  }

  function sibling(node) {
    if (node.parent) {
      for (let child of node.parent.children)
        if (child !== node) return child;
    }
    return null;
  }
}

function select_y_type() {
  tree.y_type(this.value);
  root.select('#persistence_slider').call(slider);
  localStorage.setItem('tree.y.type', this.value);
}

function select_x_type() {
  tree.x_type(this.value);
  root.select('#size_slider').call(slider);
  localStorage.setItem('tree.x.type', this.value);
}

function on_slider_change(data, range) {
  if (data.id === 'x') {
    tree.x_range(range);
    localStorage.setItem('tree.x.selection', JSON.stringify(range));
  }
  else {
    tree.y_range(range);
    localStorage.setItem('tree.y.selection', JSON.stringify(range));
  }
}

function set_persistence_range(range) {
  // if (!prevent) {
  //   prevent = true;
  //   if (saved[0] !== range[0] || saved[1] !== range[1]) {
  //     root.select('#persistence-slider')
  //       .call(slider.move, range);
  //   }
  //   prevent = false;
  // } else
  // if (prevent) console.log('tree set prevent');

}

function slider_range_update(range) {
  tree.range(range);
  if (!prevent) {
    prevent = true;
    saved = range;
    publish('persistence.range', range);
    prevent = false;
  }
  else
  if (prevent) console.log('tree slider prevent');
}

function update_feature(obj, range) {
  let section = d3.select(this.parentNode.parentNode.parentNode.parentNode);

  let feature = obj; // .feature;
  feature.filter2.range(range);
  section.select('.feature-value').text(`[${format(range[0])}, ${format(range[1])}]`);

  tree.update();
  localStorage.setItem(`feature.${feature.name}.selection`, JSON.stringify(feature.selection));
}


function activate_filter(feature) {
  feature.active = d3.select(this).property('checked');
  feature.filter2.active(feature.active);
  tree.update();
  localStorage.setItem(`feature.${feature.name}.active`, feature.active ? 'on' : 'off');
}

function update_color_by() {
  let feature = features[+this.value];
  tree.color_by(feature);
  localStorage.setItem('feature.color_by', feature.id);
}