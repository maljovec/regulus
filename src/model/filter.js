/**
 * Created by yarden on 6/6/16.
 */

function constFilter(x) { 
  return item => item === x;
}

export function noop() {
  function filter(item) { return true; }
  return filter;
}

export function not(f) {
  f = typeof f === 'function' ? f : constFilter(f);
  return item => !f(item);
}

export function and() {
  let filters = new Set();

  function filter(item) {
    for (let f of filters) {
      if (!f(item)) return false;
    }
    return true;
  }

  filter.add = function(f) {
    filters.add(f);
    return filter;
  };

  filter.delete = function(f) {
    filters.delete(f);
    return filter;
  };

  return filter;
}

export function or() {
  let filters = new Set();

  function filter(item) {
    for (let f of filters) {
      if (f(item)) return true;
    }
    return false;
  }

  filter.add = function(f) {
    filters.add(f);
    return filter;
  };

  filter.delete = function(f) {
    filters.delete(f);
    return filter;
  };

  return filter;
}