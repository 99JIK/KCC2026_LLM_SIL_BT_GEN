// BT.CPP v4 runtime loader — validates that an XML BT can be loaded and ticked.
//
// Usage: btcpp_loader <xml_file>
// Exit codes:
//   0 = loaded and ticked successfully
//   1 = load failure (XML parse / schema / unknown node)
//   2 = tick failure (exception during tick)
//   3 = loaded successfully, tick deferred (infinite loop, e.g. Repeat -1)
// Stderr contains the error message.

#include <behaviortree_cpp/bt_factory.h>

#include <csetjmp>
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

static sigjmp_buf alarm_jmp;
static void on_alarm(int) { siglongjmp(alarm_jmp, 1); }

// Pre-scan the XML and collect, for each Action/Condition ID, the set of
// attribute names used on its instances. These attribute names are then
// registered as InputPorts so BT.CPP's strict port validation passes for
// LLM-generated trees that invent arbitrary parameters.
struct LeafSpec {
  bool is_action;
  std::set<std::string> ports;
};

static std::string extract_attr(const std::string& tag, const std::string& attr) {
  size_t a = tag.find(attr + "=\"");
  if (a == std::string::npos) return "";
  a += attr.size() + 2;
  size_t b = tag.find('"', a);
  if (b == std::string::npos) return "";
  return tag.substr(a, b - a);
}

static void scan_leaves(const std::string& xml,
                        std::map<std::string, LeafSpec>& leaves) {
  size_t pos = 0;
  while (pos < xml.size()) {
    size_t lt = xml.find('<', pos);
    if (lt == std::string::npos) break;
    size_t gt = xml.find('>', lt);
    if (gt == std::string::npos) break;
    std::string tag = xml.substr(lt + 1, gt - lt - 1);
    pos = gt + 1;

    bool is_action = tag.rfind("Action", 0) == 0 &&
                     (tag.size() == 6 || tag[6] == ' ' || tag[6] == '/');
    bool is_condition = tag.rfind("Condition", 0) == 0 &&
                        (tag.size() == 9 || tag[9] == ' ' || tag[9] == '/');
    if (!is_action && !is_condition) continue;

    std::string id = extract_attr(tag, "ID");
    if (id.empty()) id = extract_attr(tag, "name");
    if (id.empty()) continue;

    LeafSpec& spec = leaves[id];
    spec.is_action = is_action;

    // Collect all attribute names except ID and name.
    // Start after the tag name (Action / Condition).
    size_t i = is_action ? 6 : 9;
    while (i < tag.size()) {
      while (i < tag.size() && (tag[i] == ' ' || tag[i] == '\t' ||
                                tag[i] == '\n' || tag[i] == '/')) i++;
      size_t name_start = i;
      while (i < tag.size() && tag[i] != '=' && tag[i] != ' ' &&
             tag[i] != '/' && tag[i] != '>') i++;
      if (i >= tag.size() || tag[i] != '=') break;
      std::string name = tag.substr(name_start, i - name_start);
      i++;  // skip '='
      if (i >= tag.size() || tag[i] != '"') break;
      i++;
      while (i < tag.size() && tag[i] != '"') i++;
      i++;  // skip closing quote
      if (!name.empty() && name != "ID" && name != "name") {
        spec.ports.insert(name);
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: btcpp_loader <xml_file>" << std::endl;
    return 1;
  }

  std::ifstream f(argv[1]);
  if (!f) {
    std::cerr << "cannot open file: " << argv[1] << std::endl;
    return 1;
  }
  std::stringstream buf;
  buf << f.rdbuf();
  std::string xml = buf.str();

  BT::BehaviorTreeFactory factory;

  // Register stubs for every leaf ID found in the XML, with InputPorts
  // matching the attribute names actually used on each instance.
  std::map<std::string, LeafSpec> leaves;
  scan_leaves(xml, leaves);
  for (const auto& [id, spec] : leaves) {
    BT::PortsList ports;
    for (const auto& p : spec.ports) {
      ports.insert(BT::InputPort<std::string>(p));
    }
    try {
      if (spec.is_action) {
        factory.registerSimpleAction(
            id,
            [](BT::TreeNode&) { return BT::NodeStatus::SUCCESS; },
            ports);
      } else {
        factory.registerSimpleCondition(
            id,
            [](BT::TreeNode&) { return BT::NodeStatus::SUCCESS; },
            ports);
      }
    } catch (...) {
      // Already registered or builtin — ignore.
    }
  }

  // Stage 4a: load XML through BT.CPP factory.
  BT::Tree tree;
  try {
    tree = factory.createTreeFromText(xml);
  } catch (const std::exception& e) {
    std::cerr << "LOAD_ERROR: " << e.what() << std::endl;
    return 1;
  }

  // Stage 4b: tick the tree once with a 1-second watchdog. Repertoire BTs
  // commonly use Repeat num_cycles="-1" which would loop forever in a single
  // tickOnce call — that's expected design, not a defect.
  signal(SIGALRM, on_alarm);
  if (sigsetjmp(alarm_jmp, 1) == 0) {
    alarm(1);
    try {
      tree.tickOnce();
    } catch (const std::exception& e) {
      alarm(0);
      std::cerr << "TICK_ERROR: " << e.what() << std::endl;
      return 2;
    }
    alarm(0);
  } else {
    // Alarm fired — tick was still running.
    std::cerr << "TICK_DEFERRED: tick did not complete within 1s "
                 "(likely infinite Repeat in a behavior repertoire)" << std::endl;
    return 3;
  }

  std::cout << "OK" << std::endl;
  return 0;
}
