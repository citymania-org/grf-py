use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::exceptions::PyRuntimeError;


fn hashed_max_overlap(d: &[u8], max_overlap: &mut Vec<u8>, overlap_ofs: &mut Vec<u16>) {
    let n = d.len();
    let mut pos: usize = 1;

    let mut prev = [0usize; 256];
    let mut hpos = vec![0usize; n];
    let mut h = d[0] ^ d[1];
    for i in 1..n - 1 {
        h = d[i - 1] ^ d[i] ^ d[i + 1];
        // h ^= d[i + 1];
        hpos[i] = prev[h as usize];
        prev[h as usize] = i;
        // h ^= d[i - 1];
    }
    prev = [0usize; 256];
    prev[(d[0] ^ d[1] ^ d[2]) as usize] = 1;
    while pos + 2 < n {
        h = d[pos] ^ d[pos + 1] ^ d[pos + 2];
        let mut max_len: usize = 2;
        let mut max_pos: usize = 0;
        let max_pattern_len = 15.min(n - pos - 1);
        let start = pos.saturating_sub((1 << 11) - 1);
        let mut i = prev[h as usize];
        while start < i {
            let mut j: usize = 0;
            while j < max_pattern_len && d[i + j] == d[pos + j + 1] { j += 1; }

            if j >= max_len {
                max_len = j + 1;
                max_pos = i - 1;
                if j == max_pattern_len { break; }
            }
            i = hpos[i];
        }
        for i in 3..=max_len {
            let end = pos + i - 1;
            if i > max_overlap[end] as usize {
                max_overlap[end] = i as u8;
                overlap_ofs[end] = (pos - max_pos) as u16;
            }
        }
        pos += 1;
        prev[h as usize] = pos;
    }
}

fn naive_max_overlap(d: &[u8], max_overlap: &mut Vec<u8>, overlap_ofs: &mut Vec<u16>) {
    let n = d.len();
    let mut pos = 1usize;
    while pos + 2 < n {
        let mut max_len: usize = 2;
        let mut max_pos: usize = 0;
        let max_pattern_len = 16.min(n - pos);
        let mut i = pos.saturating_sub((1 << 11) - 1);
        while i < pos {
            let mut j: usize = 0;
            while j < max_pattern_len && d[i + j] == d[pos + j] { j += 1; }

            if j > max_len {
                max_len = j;
                max_pos = i;
                if j == max_pattern_len { break; }
            }
            i = i.wrapping_add(1);
        }
        for i in 3..=max_len {
            let end = pos + i - 1;
            if i > max_overlap[end] as usize {
                max_overlap[end] = i as u8;
                overlap_ofs[end] = (pos - max_pos) as u16;
            }
        }
        pos += 1;
    }
}


fn buf_max_overlap(d: &[u8], max_overlap: &mut Vec<u8>, overlap_ofs: &mut Vec<u16>) {
    let n: usize = d.len();
    let mut prev = vec![0usize; 256];
    let mut buf = vec![0usize; n + 1];

    for i in 0..n {
        buf[i + 1] = prev[d[i] as usize];
        prev[d[i] as usize] = i + 1;
    }

    for l in 2..(17.min(n) as u8) {
        for pos in (1usize..n).rev() {
            buf[pos + 1] = 0;
            let start = pos.saturating_sub(1 << 11);
            let mut i = buf[pos];
            while i > start && d[i] != d[pos] {
                i = buf[i];
            }
            if i > start {
                buf[pos + 1] = i + 1;
                if l > 2 {
                    max_overlap[pos] = l;
                    overlap_ofs[pos] = (pos - i) as u16;
                }
            }
        }
    }
}


fn buf_max_overlap_unsafe(d: &[u8], max_overlap: &mut Vec<u8>, overlap_ofs: &mut Vec<u16>) {
    let n: usize = d.len();
    let mut prev = vec![0usize; 256];
    let mut buf = vec![0usize; n + 1];
    unsafe {
        for i in 0..n {
            *buf.get_unchecked_mut(i + 1) = prev[d[i] as usize];
            prev[d[i] as usize] = i + 1;
        }

        for l in 2..(17.min(n) as u8) {
            for pos in (1usize..n).rev() {
                *buf.get_unchecked_mut(pos + 1) = 0;
                let start = pos.saturating_sub(1 << 11);
                let mut i: usize = *buf.get_unchecked(pos);
                while i > start && d.get_unchecked(i) != d.get_unchecked(pos) {
                    i = *buf.get_unchecked(i);
                }
                if i > start {
                    *buf.get_unchecked_mut(pos + 1) = i + 1;
                    if l > 2 {
                        *max_overlap.get_unchecked_mut(pos) = l;
                        *overlap_ofs.get_unchecked_mut(pos) = (pos - i) as u16;
                    }
                }
            }
        }
    }
}


#[pyfunction]
fn encode_sprite_v4(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let d = data.as_bytes();
    let n = d.len();

    if n <= 3 {
        let mut res = Vec::<u8>::with_capacity(n + 1);
        res.push(n as u8);
        res.extend(d);
        return Ok(Cow::Owned(res));
    }

    let mut max_overlap = vec![0u8; n];
    let mut overlap_ofs = vec![0u16; n];

    // naive_max_overlap(&d, &mut max_overlap, &mut overlap_ofs);
    // hashed_max_overlap(&d, &mut max_overlap, &mut overlap_ofs);
    buf_max_overlap(&d, &mut max_overlap, &mut overlap_ofs);

    // let mut res = vec![0u8; 0];

    let mut open_len = vec![1u8; n];
    let mut optimal = vec![0usize; n];
    let mut prev = vec![-1isize; n];
    open_len[0] = 1;
    optimal[0] = 2;
    prev[0] = -1;
    for i in 1..n {
        optimal[i] = optimal[i - 1] + 2;
        let nl = open_len[i - 1] as usize + 1;
        let m = nl + 1 + if nl <= i { optimal[i - nl] } else { 0 };
        if m < optimal[i] {
            optimal[i] = m;
            prev[i] = -(nl as isize);
            if nl < 0x80 {
                open_len[i] = nl as u8;
            }
        }

        for j in 3..=max_overlap[i] {
            let m = optimal[i - j as usize] + 2;
            if m < optimal[i] {
                optimal[i] = m;
                prev[i] = j as isize;
            }
        }
    }

    let mut res = vec![0u8; optimal[n - 1]];
    let mut pos = optimal[n - 1];
    let mut i = n as isize - 1;

    while i >= 0 {
        let l = prev[i as usize];
        if l < 0 {
            for j in 0..-l {
                pos -= 1;
                res[pos] = d[(i - j) as usize];
            }
            pos -= 1;
            res[pos] = ((-l) & 0x7f) as u8;
            i += l;
        } else {
            pos -= 1;
            res[pos] = (overlap_ofs[i as usize] & 0xFF) as u8;
            pos -= 1;
            res[pos] = 0x80 | ((16 - l as u8) << 3) | (overlap_ofs[i as usize] >> 8) as u8;
            i -= l;
        }
    }

    Ok(Cow::Owned(res))
}


#[pyfunction]
fn encode_sprite_v3(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let d = data.as_bytes();
    let n = d.len();
    let mut res = Vec::<u8>::with_capacity(n / 3 + 5);

    if n < 3 {
        res.push(n as u8);
        res.extend(d);
        return Ok(Cow::Owned(res));
    }

    let mut prev = [0usize; 256];
    let mut hpos = vec![0usize; n];
    let mut h = d[0] ^ d[1];
    for i in 1..n - 1 {
        h ^= d[i + 1];
        hpos[i] = prev[h as usize];
        prev[h as usize] = i;
        h ^= d[i - 1]
    }
    prev = [0usize; 256];

    let mut pos: usize = 1;
    let mut literal_len: usize = 1;
    res.push(0);
    res.push(d[0]);
    // println!("d {:?}", d);
    // println!("hpos {:?}", hpos);

    prev[(d[0] ^ d[1] ^ d[2]) as usize] = 1;
    while pos < n {
        let mut max_len: usize = 0;
        let mut max_pos: usize = 0;
        if pos + 2 < n {
            h = d[pos] ^ d[pos + 1] ^ d[pos + 2];
            let max_pattern_len = 15.min(n - pos - 1);
            let start = pos.saturating_sub((1 << 11) - 1);
            let mut i = prev[h as usize];
            while start < i {
                let mut j: usize = 0;
                while j < max_pattern_len && d[i + j] == d[pos + j + 1] { j += 1; }
                // println!("M {} {}", i - 1, j + 1);
                if j >= max_len {
                    // unsafe {v2_count += 1;}
                    max_len = j + 1;
                    max_pos = i - 1;
                    if j == max_pattern_len { break; }
                }
                i = hpos[i];
            }
        }

        if max_len > 2 {
            if literal_len > 0 {
                let len_pos = res.len() - literal_len - 1;
                res[len_pos] = literal_len as u8;
                literal_len = 0;
            }
            let ofs = pos - max_pos;
            res.push(0x80 | ((16 - max_len as u8) << 3) | (ofs >> 8) as u8);
            res.push((ofs & 0xFF) as u8);
            for _ in 0..max_len {
                pos += 1;
                if pos + 2 >= n { continue; }
                prev[h as usize] = pos;
                h = d[pos] ^ d[pos + 1] ^ d[pos + 2];
            }
        } else {
            if literal_len > 0 {
                res.push(d[pos]);
                literal_len += 1;
                if literal_len == 0x80 {
                    // don't need to update len in the res as it's already 0
                    literal_len = 0;
                }
            } else {
                res.push(0);
                res.push(d[pos]);
                literal_len = 1;
            }
            pos += 1;
            prev[h as usize] = pos;
        }
    }
    if literal_len > 0 {
        let len_pos = res.len() - literal_len - 1;
        res[len_pos] = literal_len as u8;
    }

    Ok(Cow::Owned(res))
}

#[pyfunction]
fn encode_sprite_v1p(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let d = data.as_bytes();
    let mut res = Vec::<u8>::with_capacity(d.len() / 3 + 5);
    let mut literal_len: usize = 0;

    let mut pos: usize = 0;
    let n = d.len();
    let dptr = d.as_ptr();
    let mut iptr: *const u8;
    let mut iptrm: *const u8;
    let mut jptr: *const u8;
    let mut jptr_end: *const u8;
    unsafe {
    while pos < n {
        let mut max_len: usize = 2;
        let mut max_pos_ptr = dptr;
        if pos + 2 < n {
            let start: usize = pos.saturating_sub((1 << 11) - 1);
            let max_pattern_len = 16.min(n - pos);
            let posptr = dptr.add(pos);
            iptr = dptr.add(start);
            jptr_end = posptr.add(max_pattern_len);
            while iptr < posptr {
                iptrm = iptr.add(1);
                if *iptr != *posptr {
                    iptr = iptrm;
                    continue;
                }

                jptr = posptr.add(1);
                iptr = iptrm;

                while *iptrm == *jptr {
                    jptr = jptr.add(1);
                    if jptr >= jptr_end { break; }
                    iptrm = iptrm.add(1);
                }

                let j = (jptr as usize).wrapping_sub(posptr as usize);
                if j > max_len {
                    max_len = j;
                    max_pos_ptr = iptr;
                    if j == max_pattern_len { break; }
                }
            }
        }
        if max_len > 2 {
            let max_pos = (max_pos_ptr as usize).wrapping_sub(dptr as usize).wrapping_sub(1);
            if literal_len > 0 {
                let len_pos = res.len() - literal_len - 1;
                res[len_pos] = literal_len as u8;
                literal_len = 0;
            }
            let ofs = pos - max_pos;
            res.push(0x80 | ((16 - max_len as u8) << 3) | (ofs >> 8) as u8);
            res.push((ofs & 0xFF) as u8);
            pos += max_len;
        } else {
            if literal_len > 0 {
                res.push(d[pos]);
                literal_len += 1;
                if literal_len == 0x80 {
                    // don't need to update len in the res as it's already 0
                    literal_len = 0;
                }
            } else {
                res.push(0);
                res.push(d[pos]);
                literal_len = 1;
            }
            pos += 1;
        }
    }
    }
    if literal_len > 0 {
        let len_pos = res.len() - literal_len - 1;
        res[len_pos] = literal_len as u8;
    }

    Ok(Cow::Owned(res))
}

static mut v1_count: usize = 0;
static mut v2_count: usize = 0;

#[pyfunction]
fn encode_sprite_v1(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let d = data.as_bytes();
    let n = d.len();
    let mut res = Vec::<u8>::with_capacity(d.len() / 3 + 5);
    if n < 3 {
        res.push(n as u8);
        res.extend(d);
        return Ok(Cow::Owned(res));
    }

    let mut pos: usize = 1;
    let mut literal_len: usize = 1;
    res.push(0);
    res.push(d[0]);

    while pos < n {
        let mut max_len: usize = 2;
        let mut max_pos: usize = 0;
        if pos + 2 < n {
            let max_pattern_len = 16.min(n - pos);
            let mut i = pos.saturating_sub((1 << 11) - 1);
            while i < pos {
                let mut j: usize = 0;
                while j < max_pattern_len && d[i + j] == d[pos + j] { j += 1; }

                if j > max_len {
                    // unsafe {v1_count += 1;}
                    max_len = j;
                    max_pos = i;
                    if j == max_pattern_len { break; }
                }
                i = i.wrapping_add(1);
            }
        }
        if max_len > 2 {
            if literal_len > 0 {
                let len_pos = res.len() - literal_len - 1;
                res[len_pos] = literal_len as u8;
                literal_len = 0;
            }
            let ofs = pos - max_pos;
            res.push(0x80 | ((16 - max_len as u8) << 3) | (ofs >> 8) as u8);
            res.push((ofs & 0xFF) as u8);
            pos += max_len;
        } else {
            if literal_len > 0 {
                res.push(d[pos]);
                literal_len += 1;
                if literal_len == 0x80 {
                    // don't need to update len in the res as it's already 0
                    literal_len = 0;
                }
            } else {
                res.push(0);
                res.push(d[pos]);
                literal_len = 1;
            }
            pos += 1;
        }
    }
    if literal_len > 0 {
        let len_pos = res.len() - literal_len - 1;
        res[len_pos] = literal_len as u8;
    }

    Ok(Cow::Owned(res))
}

#[pyfunction]
fn encode_sprite_v2p(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let d = data.as_bytes();
    let dptr = d.as_ptr();
    let n = d.len();
    let mut res = Vec::<u8>::with_capacity(n / 3 + 5);
    let mut literal_len: usize = 0;

    let mut pos: usize = 0;
    // let mut jcount: usize = 0;
    // let mut icount: usize = 0;
    // let mut plcount: usize = 0;
    // let mut mccount: usize = 0;
    unsafe {
    while pos < n {
        let mut start = dptr.add(pos.saturating_sub((1 << 11) - 1));
        let max_pattern_len = 16.min(n - pos);
        let mut max_len: usize = 0;
        let mut max_pos: usize = 0;
        let posptr = dptr.add(pos);
        for pl in 3..=max_pattern_len {
            if pl <= max_len { continue; }
            // let end = dptr.add(pos.saturating_sub(pl - 1));
            let end = posptr;
            let mut iptr = start;
            let mut iptrm = iptr;
            let mut jptr = posptr;
            let mut iptrm_end = start; //start.add(pl);
            while iptr < end {
                iptrm = iptr.add(1);
                if *iptr != *posptr {
                    iptr = iptrm;
                    continue;
                }
                iptrm_end = iptr.add(pl);
                jptr = posptr.add(1);
                iptr = iptrm;
                while iptrm < iptrm_end && *iptrm == *jptr {
                    jptr = jptr.add(1);
                    iptrm = iptrm.add(1);
                }
                if iptrm == iptrm_end {
                    max_len = pl;
                    break;
                }
                // while iptr < end && *iptr != *posptr { iptr = iptr.add(1); }
                // iptrm_end = iptrm_end.add(1);
            }
            if max_len < pl { break; }
            // iptrm_end = iptr.add(max_pattern_len - 1).min(posptr);
            iptrm_end = iptr.add(max_pattern_len - 1);
            while iptrm < iptrm_end && *iptrm == *jptr {
                // jcount += 1;
                jptr = jptr.add(1);
                iptrm = iptrm.add(1);
            }
            // start = iptr.add(1);
            start = iptr;
            max_pos = iptr as usize - dptr as usize - 1;
            max_len = iptrm as usize - iptr as usize + 1;
        }
        if max_len > 2 {
            if literal_len > 0 {
                let len_pos = res.len() - literal_len - 1;
                res[len_pos] = literal_len as u8;
                literal_len = 0;
            }
            let ofs = pos - max_pos;
            res.push(0x80 | ((16 - max_len as u8) << 3) | (ofs >> 8) as u8);
            res.push((ofs & 0xFF) as u8);
            pos += max_len;
        } else {
            if literal_len > 0 {
                res.push(d[pos]);
                literal_len += 1;
                if literal_len == 0x80 {
                    // don't need to update len in the res as it's already 0
                    literal_len = 0;
                }
            } else {
                res.push(0);
                res.push(d[pos]);
                literal_len = 1;
            }
            pos += 1;
        }
    }
    }
    if literal_len > 0 {
        let len_pos = res.len() - literal_len - 1;
        res[len_pos] = literal_len as u8;
    }

    // println!("({}) pl={} i={} j={} if={}\n", n, plcount, icount, jcount, mccount);

    Ok(Cow::Owned(res))
}


#[pyfunction]
fn encode_sprite_v2(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let d = data.as_bytes();
    let n = d.len();
    let mut res = Vec::<u8>::with_capacity(n / 5 + 5);
    let mut literal_len: usize = 0;

    let mut pos: usize = 0;

    while pos < n {
        let mut start = pos.saturating_sub((1 << 11) - 1);
        let max_pattern_len = 16.min(n - pos);
        let mut max_len: usize = 0;
        let mut max_pos: usize = 0;
        for pl in 3..=max_pattern_len {
            if pl <= max_len { continue; }
            // let end: usize = pos.saturating_sub(pl - 1);
            let end = pos;
            let mut i = start;
            while i < end {
                let mut j: usize = 0;
                while j < pl && d[i + j] == d[pos + j] { j += 1; }
                if j >= pl {
                    while j < max_pattern_len && j + i < n && d[i + j] == d[pos + j] { j += 1; }
                    // let maxj = max_pattern_len.min(pos - i);
                    // while j < maxj && d[i + j] == d[pos + j] { j += 1; }
                    start = i.wrapping_add(1);
                    // start = i;
                    max_pos = i;
                    max_len = j;
                    break;
                }
                i = i.wrapping_add(1);
            }
            if max_len < pl { break; }
        }
        if max_len > 2 {
            if literal_len > 0 {
                let len_pos = res.len() - literal_len - 1;
                res[len_pos] = literal_len as u8;
                literal_len = 0;
            }
            let ofs = pos - max_pos;
            res.push(0x80 | ((16 - max_len as u8) << 3) | (ofs >> 8) as u8);
            res.push((ofs & 0xFF) as u8);
            pos += max_len;
        } else {
            if literal_len > 0 {
                res.push(d[pos]);
                literal_len += 1;
                if literal_len == 0x80 {
                    // don't need to update len in the res as it's already 0
                    literal_len = 0;
                }
            } else {
                res.push(0);
                res.push(d[pos]);
                literal_len = 1;
            }
            pos += 1;
        }
    }
    if literal_len > 0 {
        let len_pos = res.len() - literal_len - 1;
        res[len_pos] = literal_len as u8;
    }

    Ok(Cow::Owned(res))
}

fn find(sprite: &[u8], pat_data_pos: isize, data_pos: isize, pat_size: isize, data_size: isize) -> isize {
    for i in 0 ..= data_size - pat_size {
        let mut j = 0;
        while j < pat_size && sprite[(pat_data_pos + j) as usize] == sprite[(data_pos + i + j) as usize] {
            j += 1;
        }
        if j == pat_size {
            return i
        }
    }

    -1
}

#[pyfunction]
fn encode_sprite_truegrf(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let sprite = data.as_bytes();
    let input_size = sprite.len() as isize;

    let mut encoded = Vec::new();
    let mut literal = Vec::new();
    let mut position: isize = 0;

    while position < input_size {
        let mut start_pos = position - (1 << 11) + 1;
        if start_pos < 0 {
            start_pos = 0;
        }

        let mut max_look = input_size - position + 1;
        if max_look > 16 {
            max_look = 16;
        }

        let mut overlap_pos = 0;
        let mut overlap_len = 0;
        for i in 3..max_look {
            let result = find(sprite, position, start_pos,i, position - start_pos);
            if result < 0 {
                break;
            }

            overlap_pos = position - start_pos - result;
            overlap_len = i;
            start_pos += result;
        }

        if overlap_len > 0 {
            if !literal.is_empty() {
                encoded.push(literal.len() as u8);
                encoded.extend(literal.clone());
                literal.clear();
            }
            let val = 0x80 | ((16 - overlap_len) << 3) | (overlap_pos >> 8);
            encoded.push(val as u8);
            encoded.push(overlap_pos as u8);
            position += overlap_len;
        } else {
            literal.push(sprite[position as usize]);
            if literal.len() == 0x80 {
                encoded.push(0);
                encoded.extend(literal.clone());
                literal.clear();
            }
            position += 1;
        }
    }

    if !literal.is_empty() {
        encoded.push(literal.len() as u8);
        encoded.extend(literal.clone());
        literal.clear();
    }

    Ok(Cow::Owned(encoded))
}

#[pyfunction]
fn decode_sprite(data: &PyBytes) -> PyResult<Cow<[u8]>> {
    let mut res = Vec::<u8>::with_capacity(data.as_bytes().len());
    let mut literal_len = 0u8;
    let mut pattern_byte = 0u8;
    for &b in data.as_bytes() {
        if literal_len > 0 {
            res.push(b);
            literal_len -= 1;
        } else if pattern_byte > 0 {
            let ofs = (((pattern_byte & 7) as usize) << 8) | b as usize;
            let len = 16 - ((pattern_byte >> 3) & 15) as usize;
            if ofs > res.len() {
                return Err(PyRuntimeError::new_err("lz77 offset is outside the buffer"));
            }
            if ofs == 0 {
                return Err(PyRuntimeError::new_err("lz77 offset is zero"));
            }
            let pos = res.len() - ofs;
            // println!("OFS {} LEN {} CLEN {} POS {}\n", ofs, len, res.len(), pos);
            for i in pos..pos+len { res.push(res[i]); }
            pattern_byte = 0;
        } else {
            if b >= 0x80 { pattern_byte = b; }
            else if b > 0 { literal_len = b; }
            else { literal_len = 0x80; }
        }
    }
    Ok(Cow::Owned(res))
}

#[pyfunction]
fn print_globals() -> PyResult<()> {
    unsafe{
    println!("V1={} V2={}\n", v1_count, v2_count);}
    Ok(())
}



/// A Python module implemented in Rust.
#[pymodule]
fn grfrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_sprite, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sprite_v1, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sprite_v1p, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sprite_v2, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sprite_v2p, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sprite_v3, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sprite_v4, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sprite_truegrf, m)?)?;
    m.add_function(wrap_pyfunction!(print_globals, m)?)?;
    Ok(())
}
