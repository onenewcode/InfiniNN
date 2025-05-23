﻿use std::ops::Range;

#[derive(Clone)]
pub struct GraphTopo {
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
    pub(crate) connections: Box<[usize]>,
    pub(crate) nodes: Box<[TopoNode]>,
}

pub struct NodeRef<'a> {
    pub inputs: &'a [usize],
    pub outputs: Range<usize>,
}

#[derive(Clone)]
pub struct TopoNode {
    pub n_local: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
}

impl GraphTopo {
    /// # Safety
    ///
    /// 调用者来保证图结构的正确性
    pub const unsafe fn from_raw_parts(
        n_inputs: usize,
        n_outputs: usize,
        connections: Box<[usize]>,
        nodes: Box<[TopoNode]>,
    ) -> Self {
        Self {
            n_inputs,
            n_outputs,
            connections,
            nodes,
        }
    }

    pub const fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub const fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    pub fn n_node(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_edge(&self) -> usize {
        self.nodes
            .iter()
            .fold(self.n_inputs, |acc, n| acc + n.n_local + n.n_outputs)
    }

    pub fn global_inputs(&self) -> Range<usize> {
        0..self.n_inputs
    }

    pub fn global_outputs(&self) -> &[usize] {
        &self.connections[..self.n_outputs]
    }

    pub fn iter(&self) -> Iter {
        Iter {
            topo: self,
            i_node: 0,
            i_edge: self.n_inputs,
            i_conn: self.n_outputs,
        }
    }
}

pub struct Iter<'a> {
    topo: &'a GraphTopo,
    i_node: usize,
    i_edge: usize,
    i_conn: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = NodeRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            topo,
            i_node,
            i_edge,
            i_conn,
        } = self;
        topo.nodes.get(*i_node).map(|node| {
            let &TopoNode {
                n_local,
                n_inputs,
                n_outputs,
            } = node;
            *i_edge += n_local;
            let ans = NodeRef {
                inputs: &topo.connections[*i_conn..][..n_inputs],
                outputs: *i_edge..*i_edge + n_outputs,
            };
            *i_node += 1;
            *i_edge += n_outputs;
            *i_conn += n_inputs;
            ans
        })
    }
}
