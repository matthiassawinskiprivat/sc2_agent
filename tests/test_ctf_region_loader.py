"""
Unit tests for RegionLoader — no SC2 or MPQ files required.

Tests: XML circle parse, XML rect parse, missing region error,
cache keying, region containment checks.
"""
import pytest

from train.envs.ctf.region_loader import (
    CircleRegion,
    Point2,
    RectRegion,
    RegionLoader,
    _parse_regions_xml,
)


CIRCLE_XML = b"""
<regions>
  <region>
    <name value="circle_a"/>
    <shape type="circle">
      <center value="10.0,20.0"/>
      <radius value="5.0"/>
    </shape>
  </region>
</regions>
"""

RECT_XML_CLEAN = b"""
<regions>
  <region>
    <name value="rect_b"/>
    <shape type="rect">
      <min value="0.0,0.0"/>
      <max value="8.0,6.0"/>
    </shape>
  </region>
</regions>
"""

MIXED_XML = b"""
<regions>
  <region>
    <name value="base_a"/>
    <shape type="circle">
      <center value="50.0,50.0"/>
      <radius value="10.0"/>
    </shape>
  </region>
  <region>
    <name value="objective_b"/>
    <shape type="circle">
      <center value="100.0,100.0"/>
      <radius value="8.0"/>
    </shape>
  </region>
  <region>
    <name value="zone_rect"/>
    <shape type="rect">
      <min value="20.0,20.0"/>
      <max value="30.0,30.0"/>
    </shape>
  </region>
</regions>
"""

UNSUPPORTED_SHAPE_XML = b"""
<regions>
  <region>
    <name value="bad_shape"/>
    <shape type="polygon">
      <points value="1,2,3,4"/>
    </shape>
  </region>
</regions>
"""


class TestCircleRegion:
    def test_parse_circle_basic(self):
        regions = _parse_regions_xml(CIRCLE_XML)
        assert "circle_a" in regions
        r = regions["circle_a"]
        assert isinstance(r, CircleRegion)
        assert r.center == Point2(10.0, 20.0)
        assert r.radius == 5.0

    def test_circle_center_inside(self):
        regions = _parse_regions_xml(CIRCLE_XML)
        r = regions["circle_a"]
        assert r.contains(Point2(10.0, 20.0))

    def test_circle_boundary_inside(self):
        regions = _parse_regions_xml(CIRCLE_XML)
        r = regions["circle_a"]
        assert r.contains(Point2(15.0, 20.0))

    def test_circle_outside(self):
        regions = _parse_regions_xml(CIRCLE_XML)
        r = regions["circle_a"]
        assert not r.contains(Point2(20.0, 20.0))

    def test_circle_shape_type_property(self):
        regions = _parse_regions_xml(CIRCLE_XML)
        assert regions["circle_a"].shape_type == "circle"


class TestRectRegion:
    def test_parse_rect_basic(self):
        regions = _parse_regions_xml(RECT_XML_CLEAN)
        assert "rect_b" in regions
        r = regions["rect_b"]
        assert isinstance(r, RectRegion)
        assert r.min_corner == Point2(0.0, 0.0)
        assert r.max_corner == Point2(8.0, 6.0)

    def test_rect_center(self):
        regions = _parse_regions_xml(RECT_XML_CLEAN)
        r = regions["rect_b"]
        assert r.center == Point2(4.0, 3.0)

    def test_rect_contains_inside(self):
        regions = _parse_regions_xml(RECT_XML_CLEAN)
        r = regions["rect_b"]
        assert r.contains(Point2(4.0, 3.0))

    def test_rect_contains_corner(self):
        regions = _parse_regions_xml(RECT_XML_CLEAN)
        r = regions["rect_b"]
        assert r.contains(Point2(0.0, 0.0))

    def test_rect_outside(self):
        regions = _parse_regions_xml(RECT_XML_CLEAN)
        r = regions["rect_b"]
        assert not r.contains(Point2(9.0, 3.0))

    def test_rect_shape_type_property(self):
        regions = _parse_regions_xml(RECT_XML_CLEAN)
        assert regions["rect_b"].shape_type == "rect"


class TestMixedRegions:
    def test_parse_mixed(self):
        regions = _parse_regions_xml(MIXED_XML)
        assert len(regions) == 3

    def test_all_names_present(self):
        regions = _parse_regions_xml(MIXED_XML)
        for name in ("base_a", "objective_b", "zone_rect"):
            assert regions[name].name == name


class TestErrorCases:
    def test_unsupported_shape_raises(self):
        with pytest.raises(ValueError, match="unsupported shape type"):
            _parse_regions_xml(UNSUPPORTED_SHAPE_XML)

    def test_missing_region_raises_key_error(self):
        loader = RegionLoader()
        regions = loader.load_from_xml_bytes(MIXED_XML)
        with pytest.raises(KeyError):
            loader.get_center(regions, "nonexistent_region")

    def test_empty_xml_returns_empty_dict(self):
        regions = _parse_regions_xml(b"<regions></regions>")
        assert regions == {}


class TestRegionLoaderCache:
    def test_load_from_xml_bytes_and_cache(self):
        loader = RegionLoader()
        regions = loader.load_from_xml_bytes(MIXED_XML, cache_key="test_map")
        regions2 = loader.load_from_xml_bytes(MIXED_XML, cache_key="test_map")
        assert regions is regions2

    def test_get_cached_missing_raises(self):
        loader = RegionLoader()
        with pytest.raises(RuntimeError, match="not loaded yet"):
            loader.get_cached("nonexistent_map.sc2map")

    def test_find_units_in_circle(self):
        loader = RegionLoader()
        regions = loader.load_from_xml_bytes(CIRCLE_XML)
        reg = regions["circle_a"]
        units = [
            (1, Point2(10.0, 20.0)),
            (2, Point2(9.0, 20.0)),
            (3, Point2(20.0, 20.0)),
        ]
        inside = loader.find_units_in_region(reg, units)
        assert sorted(inside) == [1, 2]

    def test_point2_distance(self):
        a = Point2(0.0, 0.0)
        b = Point2(3.0, 4.0)
        assert abs(a.distance_to(b) - 5.0) < 1e-9
